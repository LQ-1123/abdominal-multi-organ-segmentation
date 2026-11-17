# 腹部多器官分割系统 (Abdominal Multi-Organ Segmentation)

> 课程设计项目 · 深度学习 + 强化学习 · DICOM 输入 / 三维分割 / 二维滚动查看

---

## 项目目标

用户上传一个腹部 CT 的 DICOM 序列，系统重建三维体积、自动分割多个腹部器官，
在网页里从第一层滚动到最后一层逐层查看彩色分割结果。

**核心架构：输入 3D、计算 3D、呈现 2D。**

---

## 当前进度

| 模块 | 状态 |
|---|---|
| 环境搭建 (Python 3.11 + torch 2.10 cu128 + MONAI) | ✅ 完成，sm_120 实测通过 |
| 数据集下载 | ⏳ 进行中 (TotalSeg 3GB → AMOS 22.6GB) |
| 3D U-Net 模型 (out_ch 可配置) | ✅ 完成并通过 GPU 冒烟测试 |
| 设计文档 / 标签体系 | ✅ 完成 |
| DICOM 摄取模块 | ⬜ 待开发 |
| 训练 / 评估 | ⬜ 待开发 |
| Web 查看器 | ⬜ 待开发 |
| RL 边界精修 | ⬜ 待开发 |

---

## 环境

```powershell
# 建虚拟环境（必须用 3.11，3.14 的医学包生态不完整）
py -3.11 -m venv .venv

# 安装依赖（自动处理国内网络：aria2 多线程 + 清华镜像）
.\scripts\setup_env.ps1

# 自检
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\check_env.py
```

**关键约束**：RTX 5080 是 Blackwell (sm_120)，必须用 **cu128+** 的 PyTorch，否则会报
`no kernel image is available for execution on the device`。

---

## 数据

```powershell
# 下载数据集（aria2 16 线程，支持断点续传）
.\scripts\download_data.ps1
```

| 数据集 | 内容 | 用途 |
|---|---|---|
| TotalSegmentator (small) | 多器官标注 CT | 多器官快速验证 |
| AMOS22 | 200 CT + 100 MRI，15 器官 | 多器官主实验 |
| MSD Task09 Spleen | 41 例，脾脏 | 单器官基线 / 迁移学习源域 |

标签映射集中在 [`configs/labels.yaml`](configs/labels.yaml)，
模型输出通道数与器官集合解耦，切换任务只改配置。

---

## 目录结构

```
configs/          标签体系、训练配置
data/raw/         原始数据集
data/processed/   预处理后的体积
src/
  dicom_ingest/   DICOM 解析：序列选择 / HU / 方向 / 层序 / 重采样
  models/         3D U-Net（out_ch 可配置）
  train/          训练与 5 折交叉验证
  eval/           指标计算（Dice / HD95 / ASD）
  rl/             边界精修 Agent
  render/         逐层 PNG 渲染、3D 渲染
web/              FastAPI 后端 + 滚动查看器前端
scripts/          环境安装、数据下载、自检
docs/             设计文档
```

---

## 关键设计决策

1. **呈现 2D 不等于计算 2D**：逐层独立推理会导致滚动时边界闪烁，
   因此计算必须走 3D 滑窗，只在展示层逐层渲染。
2. **内部统一中间表示**：任何输入（DICOM / NIfTI）都先归一化为 `Volume`，
   模型与输入格式彻底解耦。
3. **输出双通道**：给人看 → 网页滚动查看器 + PNG 帧流；
   给专业软件 → NIfTI 掩膜 / DICOM SEG + metrics.json。
4. **不输出"带标注的 DICOM 序列"**：那等于篡改原始影像，且体积巨大、丢失原始信息。
5. **RL 只做精细化**，不替代监督学习主干；并在 2D 上做以便可视化展示。

详见 [`docs/设计文档.md`](docs/设计文档.md)。

---

## 网络环境处理（重要）

该机器国际链路单连接被严重限速，实测：

| 目标 | 单连接 | aria2 16 线程 |
|---|---|---|
| download.pytorch.org | 627 KB/s | **24–47 MB/s** |
| pypi.tuna.tsinghua.edu.cn | — | 3.5 MB/s |
| msd-for-monai.s3 | 12 KB/s | 首段快，1.4GB 后限流 |
| zenodo.org | 7 KB/s | ~250 KB/s |

因此：**所有大文件必须走 aria2 多线程；pip 必须走国内镜像。**
