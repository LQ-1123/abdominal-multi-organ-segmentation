# Bootstrap: install remaining deps via aria2 + start dataset downloads.
# NOTE: ASCII-only on purpose (PowerShell 5.1 mis-parses UTF-8 .ps1 without BOM).
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root '.venv\Scripts\python.exe'
$logd = Join-Path $root 'outputs\logs'
$whl  = Join-Path $root 'data\wheels'
$raw  = Join-Path $root 'data\raw'
New-Item -ItemType Directory -Force -Path $logd, $whl, $raw | Out-Null

$aria = 'D:\aria2\aria2c.exe'
if (-not (Test-Path $aria)) { $aria = 'aria2c' }
$TUNA = 'https://pypi.tuna.tsinghua.edu.cn/simple'

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. torchvision
Step '1/4 torchvision wheel via aria2'
$tvName = 'torchvision-0.25.0+cu128-cp311-cp311-win_amd64.whl'
$tvPath = Join-Path $whl $tvName
if (-not (Test-Path $tvPath)) {
    & $aria -x16 -s4 -k1M --file-allocation=none --console-log-level=warn --max-tries=10 `
        --dir=$whl --out=$tvName --allow-overwrite=true --auto-file-renaming=false `
        'https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp311-cp311-win_amd64.whl'
}
if (Test-Path $tvPath) {
    Write-Host ("  wheel: {0:N1} MB" -f ((Get-Item $tvPath).Length/1MB))
    & $py -m pip install --no-cache-dir $tvPath 2>&1 |
        Tee-Object -FilePath (Join-Path $logd 'pip_tv.log') | Select-Object -Last 2
} else {
    Write-Host '  !! torchvision wheel missing, skip' -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 2. core deps
Step '2/4 core deps from TUNA mirror'
& $py -m pip install --no-cache-dir -i $TUNA `
    monai==1.6.1 pydicom SimpleITK nibabel numpy scipy scikit-image `
    opencv-python-headless matplotlib pandas pillow tqdm pyyaml `
    fastapi "uvicorn[standard]" python-multipart aiofiles 2>&1 |
    Tee-Object -FilePath (Join-Path $logd 'pip_core.log') | Select-Object -Last 3

# ---------------------------------------------------------------- 3. optional
Step '3/4 optional deps'
& $py -m pip install --no-cache-dir -i $TUNA highdicom pydicom-seg tensorboard pytest 2>&1 |
    Tee-Object -FilePath (Join-Path $logd 'pip_opt.log') | Select-Object -Last 3

# ---------------------------------------------------------------- 4. self check
Step '4/4 environment self-check'
& $py (Join-Path $root 'scripts\check_env.py')
$checkExit = $LASTEXITCODE

Write-Host "`n=== INSTALL PHASE FINISHED (check exit=$checkExit) ===" -ForegroundColor Green
& $py -m pip list 2>$null | Select-String -Pattern '^(torch|torchvision|monai|pydicom|SimpleITK|nibabel|numpy|fastapi)'
