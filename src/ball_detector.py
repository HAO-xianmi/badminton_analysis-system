"""TrackNetV2-based shuttlecock detector."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class Conv(nn.Module):
    """Conv-ReLU-BN block used by TrackNetV2-pytorch weights."""

    def __init__(self, ic, oc, bc, k=(3, 3), p="same", act=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(bc)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = x.transpose(1, 3)
        x = self.bn(x)
        return x.transpose(1, 3)


class TrackNet(nn.Module):
    """TrackNetV2 architecture compatible with ChgygLin/TrackNetV2-pytorch."""

    def __init__(self):
        super().__init__()
        self.conv2d_1 = Conv(9, 64, 512)
        self.conv2d_2 = Conv(64, 64, 512)
        self.max_pooling_1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_3 = Conv(64, 128, 256)
        self.conv2d_4 = Conv(128, 128, 256)
        self.max_pooling_2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_5 = Conv(128, 256, 128)
        self.conv2d_6 = Conv(256, 256, 128)
        self.conv2d_7 = Conv(256, 256, 128)
        self.max_pooling_3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_8 = Conv(256, 512, 64)
        self.conv2d_9 = Conv(512, 512, 64)
        self.conv2d_10 = Conv(512, 512, 64)

        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_11 = Conv(768, 256, 128)
        self.conv2d_12 = Conv(256, 256, 128)
        self.conv2d_13 = Conv(256, 256, 128)

        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_14 = Conv(384, 128, 256)
        self.conv2d_15 = Conv(128, 128, 256)

        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_16 = Conv(192, 64, 512)
        self.conv2d_17 = Conv(64, 64, 512)
        self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1, 1), padding="same")

    def forward(self, x):
        x = self.conv2d_1(x)
        x1 = self.conv2d_2(x)
        x = self.max_pooling_1(x1)

        x = self.conv2d_3(x)
        x2 = self.conv2d_4(x)
        x = self.max_pooling_2(x2)

        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x3 = self.conv2d_7(x)
        x = self.max_pooling_3(x3)

        x = self.conv2d_8(x)
        x = self.conv2d_9(x)
        x = self.conv2d_10(x)

        x = self.up_sampling_1(x)
        x = torch.concat([x, x3], dim=1)

        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.conv2d_13(x)

        x = self.up_sampling_2(x)
        x = torch.concat([x, x2], dim=1)

        x = self.conv2d_14(x)
        x = self.conv2d_15(x)

        x = self.up_sampling_3(x)
        x = torch.concat([x, x1], dim=1)

        x = self.conv2d_16(x)
        x = self.conv2d_17(x)
        x = self.conv2d_18(x)
        return torch.sigmoid(x)


class BallDetector:
    """Detect shuttlecock position from a rolling 3-frame TrackNetV2 input."""

    def __init__(self, weights_path, device="cuda", min_confidence=0.5):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"TrackNetV2 weights not found: {self.weights_path}")

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.min_confidence = float(min_confidence)
        loaded = self._load_weights()
        self.model = self._build_model(loaded)
        self.model.eval()
        self.frame_buffer = []

    def detect(self, frame):
        """Return {'pos': (x, y), 'confidence': float} or None for one frame."""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) < 3:
            return None
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        imgs = [cv2.cvtColor(cv2.resize(f, (512, 288)), cv2.COLOR_BGR2RGB) for f in self.frame_buffer]
        inp = np.concatenate(imgs, axis=2)
        inp = torch.tensor(inp).permute(2, 0, 1).float() / 255.0
        inp = inp.unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap = self._to_heatmap(self.model(inp))

        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        conf = float(heatmap[y, x])
        if conf < self.min_confidence:
            return None

        h, w = frame.shape[:2]
        x_orig = int(x / 512 * w)
        y_orig = int(y / 288 * h)
        return {"pos": (x_orig, y_orig), "confidence": conf}

    def _build_model(self, loaded):
        """Return a callable model from either a full model or a state dict."""
        if hasattr(loaded, "eval"):
            return loaded.to(self.device)

        if isinstance(loaded, dict) and "model" in loaded and hasattr(loaded["model"], "eval"):
            return loaded["model"].to(self.device)

        state_dict = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else None
        if not isinstance(state_dict, dict):
            raise TypeError(
                "TrackNetV2 weights did not load as a model or state dict."
            )

        model = TrackNet().to(self.device)
        model.load_state_dict(state_dict)
        return model

    def _load_weights(self):
        try:
            return torch.load(str(self.weights_path), map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(str(self.weights_path), map_location=self.device)
        except Exception:
            return torch.load(str(self.weights_path), map_location=self.device, weights_only=False)

    def _to_heatmap(self, output):
        """Normalize common TrackNet output shapes to a 2-D numpy heatmap."""
        if isinstance(output, (list, tuple)):
            output = output[0]
        if output.ndim == 4:
            output = output[0, -1]
        elif output.ndim == 3:
            output = output[-1]
        elif output.ndim != 2:
            raise ValueError(f"Unsupported TrackNet output shape: {tuple(output.shape)}")
        return output.detach().cpu().numpy()
