# Persistent downloader with stall recovery.
#
# Why: the uplink shows a burst quota (~2.5-3 GB at 20-40 MB/s) then drops to
# ~12 KB/s. A single aria2 run therefore dies near completion. This script
# loops: let aria2 run, abort when speed collapses, sleep to let the quota
# refill, resume (aria2 --continue). Idempotent and safe to re-run.
#
# ASCII-only on purpose (PowerShell 5.1 + UTF-8 no BOM = parse errors).

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$raw = Join-Path $root 'data\raw'
$logd = Join-Path $root 'outputs\logs'
New-Item -ItemType Directory -Force -Path $raw, $logd | Out-Null

$aria = 'D:\aria2\aria2c.exe'
if (-not (Test-Path $aria)) { $aria = 'aria2c' }
$log = Join-Path $logd 'fetch.log'

function Note($m) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Get-RemoteSize($url) {
    $h = curl.exe -sI -L --max-time 30 $url 2>$null
    $cl = ($h | Select-String -Pattern '^content-length:') -replace '[^\d]', ''
    if ($cl) { return [long]$cl } else { return 0 }
}

function Ensure($url, $out, $label, $maxRounds = 80, $coolDownSec = 75) {
    $target = Join-Path $raw $out
    $remote = Get-RemoteSize $url
    if ($remote -le 0) { Note "WARN $label : cannot read remote size, will try anyway" }
    Note ("TARGET $label  remote={0:N0} MB" -f ($remote / 1MB))

    for ($round = 1; $round -le $maxRounds; $round++) {
        $have = 0
        if (Test-Path $target) { $have = (Get-Item $target).Length }
        if ($remote -gt 0 -and $have -eq $remote) {
            Note ("COMPLETE {0} ({1:N0} MB)" -f $label, ($have / 1MB))
            return $true
        }
        Note ("round {0}: {1} have={2:N0} MB / {3:N0} MB" -f $round, $label, ($have / 1MB), ($remote / 1MB))

        # Real sustained throughput to Zenodo is ~200 KB/s with 16 connections
        # (no CDN edge in CN). Threshold must sit BELOW that, otherwise aria2
        # aborts every attempt and nothing progresses. 20K only kills truly
        # dead connections.
        & $aria -c -x16 -s16 -k1M --file-allocation=none `
            --lowest-speed-limit=20K --max-tries=3 --retry-wait=10 `
            --timeout=60 --connect-timeout=30 --console-log-level=warn --summary-interval=0 `
            --dir=$raw --out=$out --allow-overwrite=true --auto-file-renaming=false $url

        $now = 0
        if (Test-Path $target) { $now = (Get-Item $target).Length }
        if ($remote -gt 0 -and $now -eq $remote) {
            Note ("COMPLETE {0} ({1:N0} MB)" -f $label, ($now / 1MB))
            return $true
        }
        if ($now -le $have) { Note "  no progress this round; longer cool-down" ; $coolDownSec = [Math]::Min($coolDownSec * 1.5, 300) }
        Note ("  cooling down {0}s" -f $coolDownSec)
        Start-Sleep -Seconds $coolDownSec
    }
    Note "GIVEUP $label"
    return $false
}

Note '===== persistent fetcher start ====='

# 1) finish TotalSegmentator small (multi-organ, ~190 MB left)
Ensure 'https://zenodo.org/api/records/10047263/files/Totalsegmentator_dataset_small_v201.zip/content' `
       'Totalsegmentator_small.zip' 'TotalSeg-Small(3.0GB)'

# 2) AMOS22 (15 organs, 200 CT + 100 MRI) - the main multi-organ dataset
Ensure 'https://zenodo.org/api/records/7155725/files/amos22.zip/content' `
       'amos22.zip' 'AMOS22(22.6GB)' 400

# 3) MSD Spleen (single-organ baseline / transfer source)
Ensure 'https://msd-for-monai.s3.amazonaws.com/Task09_Spleen.tar' `
       'Task09_Spleen.tar' 'MSD-Spleen(1.5GB)'

Note '===== persistent fetcher done ====='
Get-ChildItem $raw -File | ForEach-Object { Note ("  {0,-34} {1,8:N2} GB" -f $_.Name, ($_.Length / 1GB)) }
