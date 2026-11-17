# 交接文档（HANDOVER）

> 课程：数字图像处理大作业 · 图像分割方向（腹部 CT 多器官分割）
> 本仓库包含全部代码、实验记录、报告图表与文档。**数据集与训练权重不入库**，按下文链接获取，下载放回原位即可直接接力。

---

## 1. 五分钟接手上手

```powershell
git clone <本仓库地址>
cd abdominal-multi-organ-segmentation

# ① 环境（必须 Python 3.11）
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# GPU 注意：RTX 5080(Blackwell/sm_120) 必须装 cu128 版 torch；老显卡装对应版本即可，代码为纯 PyTorch + MONAI

# ② 数据集（见第 2 节链接，下载后放入 data/raw/）
# ③ 训练权重（见第 3 节，从 Release 下载放回 outputs_v2/fold*/best.pth）

# ④ 自检
$env:PYTHONIOENCODING='utf-8'
python scripts/check_env.py
python -m src.data.amos_dataset          # 应输出可用病例数
```

数据缓存构建（下载 AMOS22 并解压到 `data/processed/amos22/` 后执行）：

```powershell
python scripts/build_cache.py --workers 6   # 重采样到 1.5mm 等向 + npy 缓存，约 22GB
```

## 2. 数据集下载（不入库）

| 数据集 | 用途 | 大小 | 下载 |
|---|---|---|---|
| **AMOS22**（主实验） | 200 CT + 100 MRI，15 器官 | 22.6GB | `https://zenodo.org/api/records/7155725/files/amos22.zip/content` |
| TotalSegmentator small | 多器官补充/备选 | 3.0GB | `https://zenodo.org/api/records/10047263/files/Totalsegmentator_dataset_small_v201.zip/content` |
| MSD Task09 Spleen | 单器官基线 | 1.5GB | `https://msd-for-monai.s3.amazonaws.com/Task09_Spleen.tar` |

- 一键脚本：`.\scripts\download_data.ps1`（aria2 16 线程断点续传；国际链路慢时强烈建议用）
- 引用信息：AMOS22 — Ji et al., "AMOS: A Large-Scale Abdominal Multi-Organ Benchmark", MICCAI 2022（写报告参考文献用）
- AMOS 解压后目录约定：`data/processed/amos22/`（`images_ct/`、`labels_ct/` 等）

## 3. 训练权重下载（不入库）

从本仓库的 **GitHub Release（tag: `handover`）** 下载，放回对应路径：

| Release 资产 | 放回位置 | 说明 |
|---|---|---|
| `v2_fold0_best.pth` ... `v2_fold4_best.pth` | `outputs_v2/fold<k>/best.pth` | **V2 模型（主力）**，5 折 mDice 0.8056 |
| `v1_fold0_best.pth` ... `v1_fold4_best.pth` | `outputs/fold<k>/best.pth` | V1 基线，mDice 0.7665（对照用，可选） |

```powershell
# 一键下载并放回（项目根目录执行；Release 页面的说明里也有同款命令）
gh release download handover --repo LQ-1123/abdominal-multi-organ-segmentation
foreach ($k in 0..4) {
  Move-Item "v2_fold${k}_best.pth" "outputs_v2/fold$k/best.pth" -Force
  if (Test-Path "v1_fold${k}_best.pth") { Move-Item "v1_fold${k}_best.pth" "outputs/fold$k/best.pth" -Force }
}
```
（Release 里若看到一个名为 `best.pth` 的多余文件，是上传测试残留，可忽略。）

每个 `best.pth` 约 87MB（含 optimizer 状态）。**做消融实验只需 outputs_v2 的 5 个文件。**

## 4. 仓库里有什么

```
src/               全部核心代码：数据管线 / 3D U-Net / 训练 / 损失 / 推理 / 后处理 / RL / 渲染 / DICOM 摄取
scripts/           环境安装、数据下载与缓存、训练入口、评估汇总、报告图表、基准测试
configs/labels.yaml 器官标签体系
web/               FastAPI 后端 + 前端滚动查看器
docs/设计文档.md    系统设计（报告「系统设计」章节底稿）
docs/分工表.md      已完成工作清单（截至 2026-11-17，负责人：孙钰林）
docs/下一步任务书.md 剩余任务 P0/P1/P2 + 排期 ← **接手先读这个**
outputs*/summary.json、cv_compare.csv、train_log.csv、val_organ_dice.json   全部实验数字（无需重训即可引用）
outputs/figs/      报告/PPT 用图（训练曲线、分器官 Dice 等）
outputs/demo/      3D 渲染与逐层可视化样例
```

## 5. 当前状态（一句话）

深度学习部分已完备：AMOS22 上 3D U-Net 5 折交叉验证，V1→V2 mDice 0.767→0.806；**剩余核心工作是作业硬性要求的「课本传统图像处理方法 + 对照消融实验」与报告/PPT**，明细见 `docs/下一步任务书.md`。

## 6. 小显存机器注意（4GB 笔记本实测可行）

- 推理用 `Predictor(..., sw_batch_size=1)`（默认 2 会多占约 500MB）；实测 96³ patch 峰值约 2.4GB。
- 微调用 `--patch 96 --batch 1` + AMP，峰值约 1.1GB。
- 唯一放不下的是复现完整训练（128³ batch2，峰值 5.1GB）——不需要重跑，权重已给。
- 验证脚本：`scripts/bench_vram.py`、`scripts/bench_real_infer.py`。
- 老 GPU（Pascal 及更早）需装旧版 cu121 的 torch，cu128 轮子不支持。

## 7. 常见坑（原开发环境踩过）

1. **中文路径**：SimpleITK 在 Windows 下按 ANSI 读路径会报错，代码里已用 nibabel 绕开——不要改回 `sitk.ReadImage`。
2. **层序排序**：DICOM 必须按 `ImagePositionPatient` 法向投影排，不能用 `InstanceNumber`。
3. **标签插值**：重采样标签必须最近邻，线性插值会产生不存在的类别值。
4. **5 折划分**：必须按病例划分（`make_folds`，seed=42），按 patch 划分会数据泄漏。
5. **HU 窗**：训练侧 [-150, 250]（`amos_dataset.py`），与渲染用的窗宽窗位是两回事。
