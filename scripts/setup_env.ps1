# 医学图像分割 - 环境搭建脚本 (PowerShell 5.1 兼容)
# 策略：
#   1) PyTorch cu128 wheel 用 aria2 多线程下载（单连接被限速，16 连接可达 24MB/s）
#   2) 其余包走清华 PyPI 镜像（3.5MB/s）
# 说明：RTX 5080 是 sm_120 (Blackwell)，必须 cu128+ 版本

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root '.venv\Scripts\python.exe'
$logd = Join-Path $root 'outputs\logs'
$whl  = Join-Path $root 'data\wheels'
New-Item -ItemType Directory -Force -Path $logd, $whl | Out-Null

$aria = 'D:\aria2\aria2c.exe'
if (-not (Test-Path $aria)) { $aria = 'aria2c' }

function Log($msg, $color = 'Cyan') { Write-Host $msg -ForegroundColor $color }

function Step($title, [scriptblock]$body) {
    $t0 = Get-Date
    Log "`n===== $title ====="
    & $body
    Write-Host ("  [DONE] {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
}

# ---------- 1. aria2 多线程拉取 torch wheel ----------
Step "下载 PyTorch cu128 wheel (aria2 x16)" {
    $torchWhl = 'torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl'
    $torchUrl = "https://download.pytorch.org/whl/cu128/$torchWhl"
    $local    = Join-Path $whl 'torch-2.10.0+cu128-cp311-cp311-win_amd64.whl'
    if (Test-Path $local) {
        Log "  已存在，跳过"
    } else {
        & $aria -x16 -s16 -k1M --file-allocation=none --console-log-level=warn `
            --dir=$whl --out='torch-2.10.0+cu128-cp311-cp311-win_amd64.whl' --allow-overwrite=true $torchUrl
    }
    if (Test-Path $local) { "  torch wheel: {0:N2} GB" -f ((Get-Item $local).Length/1GB) }
}

# ---------- 2. 安装 torch ----------
Step "安装 torch 2.10.0+cu128" {
    $local = Join-Path $whl 'torch-2.10.0+cu128-cp311-cp311-win_amd64.whl'
    & $py -m pip install --no-cache-dir $local 2>&1 |
        Tee-Object -FilePath (Join-Path $logd 'pip_torch.log') | Select-Object -Last 6
}

# ---------- 3. 其余依赖走清华镜像 ----------
$TUNA = 'https://pypi.tuna.tsinghua.edu.cn/simple'

Step "torchvision (cu128 源)" {
    & $py -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 `
        torchvision 2>&1 | Tee-Object -FilePath (Join-Path $logd 'pip_tv.log') | Select-Object -Last 4
}

Step "医学影像与科学计算 (清华镜像)" {
    & $py -m pip install --no-cache-dir -i $TUNA `
        monai==1.6.1 pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg pylibjpeg-rle SimpleITK nibabel numpy scipy scikit-image `
        opencv-python-headless matplotlib pandas pillow tqdm pyyaml 2>&1 |
        Tee-Object -FilePath (Join-Path $logd 'pip_core.log') | Select-Object -Last 6
}

Step "Web 与可选组件 (清华镜像)" {
    & $py -m pip install --no-cache-dir -i $TUNA `
        fastapi "uvicorn[standard]" python-multipart jinja2 aiofiles `
        highdicom pydicom-seg tensorboard pytest 2>&1 |
        Tee-Object -FilePath (Join-Path $logd 'pip_web.log') | Select-Object -Last 6
}

# ---------- 4. 自检 ----------
Step "环境自检" { & $py (Join-Path $root 'scripts\check_env.py') }

Log "`n全部完成。" 'Green'
