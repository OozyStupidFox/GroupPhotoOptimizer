# 全家福照片优化器

下载前确保电脑已安装Git LFS。Git LFS下载链接：https://git-lfs.com/

复制此代码克隆此项目工程：git clone https://github.com/OozyStupidFox/GroupPhotoOptimizer.git

这个工程面向机位基本不变、人物站位相同的一组连拍照片。它会在本机完成以下工作：

1. 对 6240×4160 等高分辨率照片分块检测人脸，并计算清晰度和严格的人脸特征。
2. 配准所有照片，用“站位 + 人脸特征模板”追踪每位亲属，分析闭眼、单眼眨眼和嘴部异常。
3. 选择表情正常人数最多的照片作为母片，只从同一个人的其他照片中替换眼周或嘴周。
4. 在替换前再次检查身份、清晰度和非替换脸部区域的结构相似度。任一检查失败便保留原图。

所有家庭照片都只在本机处理。首次运行会从 OpenCV 和 MediaPipe 官方地址下载约 50 MB 的模型文件，模型下载后可以离线运行。

## 环境与安装

支持 Python 3.8–3.12。Windows PowerShell 中运行：

```powershell
py -3 -m pip install -e .
py -3 -m group_photo_optimizer download-models
```

照片放在 `images/` 中，支持 JPG、PNG、TIFF 和 BMP。输入照片应满足：

- 分辨率和方向一致；
- 相机位置基本不变；
- 人物站位不变，头部小幅移动可以接受；
- 不要把导出的 `output/final.jpg` 再放回 `images/`。

## 运行

先只做检测、评分和母片选择：

```powershell
py -3 -m group_photo_optimizer run --analyze-only
```

打开 `output/report.html`，对照 `output/base_annotated.jpg` 检查编号。绿色框表示通过，红色框后的 `E` 表示眼睛异常，`M` 表示嘴部异常。

确认分析合理后执行完整合成。`--reuse-analysis` 会直接复用人脸分析缓存：

```powershell
py -3 -m group_photo_optimizer run --reuse-analysis
```

主要输出：

| 文件 | 内容 |
| --- | --- |
| `output/final.jpg` | 保留母片 EXIF/ICC、以 JPEG 98 质量导出的成片 |
| `output/report.html` | 母片评分和每一次替换/跳过的审核报告 |
| `output/base_annotated.jpg` | 人物编号及异常区域标注 |
| `output/replacement_review.jpg` | 成功替换的母片/供体/结果放大对照 |
| `output/observations.csv` | 每个人在每张照片中的原始分数 |
| `output/audit.json` | 便于程序读取的照片评分与替换记录 |
| `output/analysis_cache.pkl` | 本机分析缓存，不应接收来源不明的缓存文件 |

## 人工修正

自动表情判断不可能覆盖每位老人、婴儿和遮挡情形。根据标注图修改 `config.yaml` 的 `manual` 部分，再用缓存重跑：

```yaml
manual:
  base_image: DSCF1254.JPG
  exclude_tracks: [P012, P047]
  force_donors:
    P023:
      eyes: DSCF1258.JPG
      mouth: DSCF1260.JPG
```

- `base_image`：手工指定母片。留空时自动选择。
- `exclude_tracks`：这些人完全不做自动替换。
- `force_donors`：只尝试指定照片，但指定候选仍必须通过所有安全阈值。

改 `manual` 和 `replacement` 参数不会使分析缓存失效。改检测、配准、追踪或评分参数后，需要去掉 `--reuse-analysis` 重新分析。

## 严格身份策略

本工程不按“最像的人”全局搜索。照片先进行背景特征配准，候选必须处在同一站位附近，再同时通过以下检查：

- 与母片中的该人达到 `tracking.replacement_similarity`；
- 与该人在整组照片中的人脸特征模板达到同一阈值；
- 非替换的脸部区域达到 `replacement.min_outside_similarity`；
- 候选区域表情通过、检测可信且清晰度不明显下降。
- 供体闭眼/嘴部异常绝对分数低于上限，并比母片至少改善 `min_expression_improvement`。

默认阈值针对亲属相似的情况设置得较保守。若报告中大量候选因身份相似度不足而跳过，可以小幅降低 `replacement_similarity`，建议每次不超过 0.02，并逐个检查结果。不要为了增加替换数量一次性大幅降低阈值。

## 测试

```powershell
py -3 -m unittest discover -s tests -v
```

最终照片应在 100% 放大下检查所有红框人物。该工具只替换已有照片中的局部真实五官，不会生成不存在的表情；没有可靠候选时会明确跳过。

## 单文件 EXE

已构建版本位于 `dist/GroupPhotoOptimizer.exe`。在其他 Windows 电脑上使用时：

1. 把 EXE 放入一个可写目录。
2. 在同一目录建立 `images` 文件夹并放入连拍照片。
3. 双击 EXE 打开桌面界面，在“图片文件夹”中选择照片目录并点击“开始处理”。首次运行会在同目录生成 `config.yaml`，模型已包含在 EXE 内，不需要 Python 或网络。
4. 成片、报告和日志写入同目录的 `output` 文件夹。

命令行运行可以复用分析缓存：

```powershell
.\GroupPhotoOptimizer.exe run --reuse-analysis
```

每次运行生成 `output/run_YYYYMMDD_HHMMSS.log`，`output/latest.log` 始终指向最近一次日志的副本。日志包含：

- 所有照片的质量排名、通过人脸数、睁眼人数、关键点人数和质量分数；
- 成功替换的人物编号、区域、供体照片和两项相似度；
- 找不到合适供体的人物编号、基础表情分数、候选淘汰原因；
- 每张照片的检测耗时，以及模型、检测、配准、追踪、替换、输出和总运行时长；
- 运行失败时的完整错误堆栈。

桌面界面的“实时日志”页会同步显示这些内容；处理结束后自动切换到“审核报告”页，在同一窗口内展示 `report.html`、标注图、替换对照和审计表。

重新构建 EXE：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```
