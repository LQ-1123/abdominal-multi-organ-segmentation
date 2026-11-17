# Chunked parallel downloader (robust replacement for aria2 resume).
#
# Why: aria2's control file got corrupted after repeated aborts and then
# re-requested the same stuck pieces forever (0 progress), while independent
# curl range requests to the same URL ran at ~250 KB/s. So: download N
# independent byte ranges into N part files, then concatenate. Every part is
# resumable on its own; a part file whose size already matches is skipped.
#
# Measured ceiling on this link: ~250 KB/s aggregate for international hosts.
# More connections do NOT bypass it (16 conn=200 KB/s, 24 conn=246 KB/s).
#
# ASCII-only on purpose (PowerShell 5.1 + UTF-8 no BOM = parse errors).

param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Parts = 24,
    [long]$StartOffset = 0,   # >0 = only fetch the tail (append to existing file)
    [string]$Label = 'file'
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$raw = Join-Path $root 'data\raw'
$partDir = Join-Path $raw ("parts_" + [IO.Path]::GetFileNameWithoutExtension($Out))
$logd = Join-Path $root 'outputs\logs'
New-Item -ItemType Directory -Force -Path $raw, $partDir | Out-Null

$log = Join-Path $logd 'chunked.log'
function Note($m) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

# ---------- remote size ----------
$h = curl.exe -sI -L --max-time 40 $Url 2>$null
$total = [long](($h | Select-String -Pattern '^content-length:') -replace '[^\d]', '')
if ($total -le 0) { Note "FATAL $Label : cannot read content-length"; exit 1 }
Note ("TARGET {0} : {1:N0} MB, offset={2:N0}, parts={3}" -f $Label, ($total/1MB), $StartOffset, $Parts)

$span = $total - $StartOffset
if ($span -le 0) { Note "nothing to do (already complete)"; exit 0 }
$seg = [long][Math]::Ceiling($span / $Parts)

# ---------- per-part resumable fetch ----------
# NOTE: curl forbids -C (resume) together with -r (range). So resume is done
# manually: fetch the missing tail of a part into a .rest file, then append.
$procs = @()
for ($i = 0; $i -lt $Parts; $i++) {
    $s = $StartOffset + $i * $seg
    $e = [Math]::Min($s + $seg - 1, $total - 1)
    if ($s -gt $total - 1) { continue }
    $partFile = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
    $want = $e - $s + 1

    $have = 0
    if (Test-Path $partFile) {
        $have = (Get-Item $partFile).Length
        if ($have -eq $want) { continue }          # this part is finished
        if ($have -gt $want) { Remove-Item $partFile -Force; $have = 0 }
    }
    $from = $s + $have
    $restFile = Join-Path $partDir ("part_{0:D3}.rest" -f $i)
    $procs += Start-Process -FilePath 'curl.exe' -ArgumentList @(
        '-s', '-L', '--retry', '30', '--retry-delay', '5', '--retry-all-errors',
        '-r', "$from-$e",
        '-o', $restFile,
        '--max-time', '86400',
        $Url
    ) -PassThru -NoNewWindow
}
Note "started $($procs.Count) curl workers"

# ---------- wait ----------
$t0 = Get-Date
while (($procs | Where-Object { -not $_.HasExited }).Count -gt 0) {
    Start-Sleep -Seconds 20
    $done = (Get-ChildItem "$partDir\part_*.bin" -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $el = ((Get-Date) - $t0).TotalSeconds
    if ($el -gt 0) {
        Note ("  progress {0:N0} / {1:N0} MB  ({2:N0} KB/s)" -f ($done/1MB), ($span/1MB), ($done/$el/1KB))
    }
    if ($el -gt 90000) { Note 'hard timeout'; break }
}

# ---------- merge .rest tails into their part files ----------
Get-ChildItem "$partDir\part_*.rest" -ErrorAction SilentlyContinue | ForEach-Object {
    $rest = $_.FullName
    $bin = $rest -replace '\.rest$', '.bin'
    if (Test-Path $bin) {
        $dst = [IO.File]::Open($bin, [IO.FileMode]::Append, [IO.FileAccess]::Write)
        $src = [IO.File]::OpenRead($rest)
        $src.CopyTo($dst, 1048576)
        $src.Close(); $dst.Close()
    } else {
        Move-Item $rest $bin -Force
    }
    Remove-Item $rest -Force -ErrorAction SilentlyContinue
}

# ---------- verify every part ----------
$missing = 0
for ($i = 0; $i -lt $Parts; $i++) {
    $s = $StartOffset + $i * $seg
    $e = [Math]::Min($s + $seg - 1, $total - 1)
    if ($s -gt $total - 1) { continue }
    $want = $e - $s + 1
    $partFile = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
    $got = if (Test-Path $partFile) { (Get-Item $partFile).Length } else { 0 }
    if ($got -ne $want) { Note ("  PART INCOMPLETE {0:D3}: {1}/{2}" -f $i, $got, $want); $missing++ }
}
if ($missing -gt 0) { Note "FAIL $Label : $missing part(s) incomplete; re-run to resume"; exit 2 }

# ---------- concatenate ----------
$target = Join-Path $raw $Out
Note "concatenating $Parts parts -> $Out"
$fs = [IO.File]::Open($target, [IO.FileMode]::Append, [IO.FileAccess]::Write)
for ($i = 0; $i -lt $Parts; $i++) {
    $s = $StartOffset + $i * $seg
    if ($s -gt $total - 1) { continue }
    $partFile = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
    $src = [IO.File]::OpenRead($partFile)
    $src.CopyTo($fs, 1048576)
    $src.Close()
}
$fs.Close()

$final = (Get-Item $target).Length
Note ("DONE {0} : {1:N0} bytes (expect {2:N0})" -f $Label, $final, $total)
if ($final -eq $total) { Note 'SIZE OK -> removing part files'; Remove-Item "$partDir\*" -Force -ErrorAction SilentlyContinue }
else { Note 'SIZE MISMATCH - keep parts for inspection' }
