# Dataset download: fresh multi-connection downloads (resume proved unreliable on S3).
# NOTE: ASCII-only on purpose (PowerShell 5.1 + UTF-8 without BOM = parse errors).
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$raw  = Join-Path $root 'data\raw'
$logd = Join-Path $root 'outputs\logs'
New-Item -ItemType Directory -Force -Path $raw, $logd | Out-Null

$aria = 'D:\aria2\aria2c.exe'
if (-not (Test-Path $aria)) { $aria = 'aria2c' }
$log = Join-Path $logd 'download.log'

function Note($m) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Fetch($url, $outName, $label) {
    $target = Join-Path $raw $outName
    Note "START $label -> $outName"
    # fresh download: drop stale partial + control file
    foreach ($f in @($target, "$target.aria2")) {
        if (Test-Path $f) { Remove-Item $f -Force -ErrorAction SilentlyContinue }
    }
    & $aria -x16 -s16 -k1M --file-allocation=none --console-log-level=warn `
        --max-tries=20 --retry-wait=8 --timeout=60 --connect-timeout=30 --summary-interval=0 `
        --dir=$raw --out=$outName --allow-overwrite=true --auto-file-renaming=false $url
    if (Test-Path $target) {
        Note ("DONE  {0} : {1:N2} GB" -f $label, ((Get-Item $target).Length/1GB))
    } else {
        Note "FAIL  $label"
    }
}

Note '===== download queue start ====='

# Order rationale: S3 throttles a single object after ~1.4GB (observed twice),
# so MSD goes last. Multi-organ datasets first -- they are what the project needs.

# 1) TotalSegmentator small 3.0GB - multi-organ
Fetch 'https://zenodo.org/api/records/10047263/files/Totalsegmentator_dataset_small_v201.zip/content' `
      'Totalsegmentator_small.zip' 'TotalSeg-Small(3.0GB)'

# 2) AMOS22 22.6GB - multi-organ main dataset (15 organs, 200 CT + 100 MRI)
Fetch 'https://zenodo.org/api/records/7155725/files/amos22.zip/content' 'amos22.zip' 'AMOS22(22.6GB)'

# 3) MSD Spleen 1.5GB - single organ baseline (retry last; S3 rate-limit prone)
Fetch 'https://msd-for-monai.s3.amazonaws.com/Task09_Spleen.tar' 'Task09_Spleen.tar' 'MSD-Spleen(1.5GB)'

Note '===== download queue end ====='
Get-ChildItem $raw -File | ForEach-Object { Note ("  {0,-34} {1,8:N2} GB" -f $_.Name, ($_.Length/1GB)) }
