# 客户快速交付流程

1. 安装 Git、Python 3.10+ 和 FFmpeg。
2. 克隆项目并执行安装脚本:

```powershell
git clone https://github.com/HAO-xianmi/badminton_analysis-system.git
cd badminton_analysis-system
.\scripts\setup.ps1
```

3. 把比赛视频放入 `data/raw/`。
4. 执行:

```powershell
.\.venv\Scripts\Activate.ps1
python src\add_source.py
```

5. 按窗口提示完成球场四角标定和比分区域选择。
6. 执行:

```powershell
python src\main.py --source <视频源名称> --output data\output\result.mp4
```

7. 输出文件在 `data/output/result.mp4`。

如需先试跑 30 秒:

```powershell
python src\main.py --source <视频源名称> --output data\output\sample_30s.mp4 --duration 30
```
