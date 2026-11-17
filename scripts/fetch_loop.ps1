# Segmented downloader with periodic connection recycling.
#
# Observed behaviour on this link (direct and via TUN VPN, with and without
# aria2): throughput is high for the first ~60-90 s, then the established
# connections get throttled to ~0 while NEW connections remain fast. So no
# single long-lived transfer ever finishes. Fix: download in short rounds --
# spawn fresh connections, transfer for ~75 s, kill them, resume from the
# partial part files, repeat. Each part file is appended safely because every
# round continues exactly at (part_start + bytes_already_on_disk).
#
# ASCII-only on purpose (PowerShell 5.1 + UTF-8 no BOM = parse errors).

param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Parts = 24,
    [int]$RoundSeconds = 75,
    [int]$MaxRounds = 80
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$raw = Join-Path $root 'data\raw'
$partDir = Join-Path $raw ("parts_" + [IO.Path]::GetFileNameWithoutExtension($Out))
$logd = Join-Path $root 'outputs\logs'
New-Item -ItemType Directory -Force -Path $raw, $partDir, $logd | Out-Null
$log = Join-Path $logd 'loopdl.log'

function Note($m) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

# ---- remote size ----
$h = curl.exe -sI -L --max-time 40 $Url 2>$null
$total = [long](($h | Select-String -Pattern '^content-length:') -replace '[^\d]', '')
if ($total -le 0) { Note 'FATAL: cannot read content-length'; exit 1 }
$seg = [long][Math]::Ceiling($total / $Parts)
Note ("START {0}  total={1:N0} MB  parts={2}" -f $Out, ($total / 1MB), $Parts)

function Get-PartRange($i) {
    $s = $i * $seg
    $e = [Math]::Min($s + $seg - 1, $total - 1)
    return @($s, $e)
}

function Merge-RestInto($i) {
    # append any '.rest' tail left by the previous round to the part file
    $bin = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
    $rest = Join-Path $partDir ("part_{0:D3}.rest" -f $i)
    if (Test-Path $rest) {
        if ((Get-Item $rest).Length -gt 0) {
            if (Test-Path $bin) {
                $dst = [IO.File]::Open($bin, [IO.FileMode]::Append, [IO.FileAccess]::Write)
            } else {
                $dst = [IO.File]::Create($bin)
            }
            $src = [IO.File]::OpenRead($rest)
            $src.CopyTo($dst, 1048576)
            $src.Close(); $dst.Close()
        }
        Remove-Item $rest -Force -ErrorAction SilentlyContinue
    }
}

for ($round = 1; $round -le $MaxRounds; $round++) {
    # --- normalise part files ---
    for ($i = 0; $i -lt $Parts; $i++) { Merge-RestInto $i }

    # --- how much do we already have? ---
    $have = [long]0
    for ($i = 0; $i -lt $Parts; $i++) {
        $r = Get-PartRange $i
        $want = $r[1] - $r[0] + 1
        $bin = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
        $got = if (Test-Path $bin) { [long](Get-Item $bin).Length } else { [long]0 }
        if ($got -gt $want) { $got = $want }
        $have += $got
    }
    $pct = 100.0 * $have / $total
    Note ("round {0}: {1:N0}/{2:N0} MB ({3:N1}%)" -f $round, ($have / 1MB), ($total / 1MB), $pct)
    if ($have -ge $total) { break }

    # --- spawn fresh connections for every incomplete part ---
    $procs = @()
    for ($i = 0; $i -lt $Parts; $i++) {
        $r = Get-PartRange $i
        $s = $r[0]; $e = $r[1]
        $want = $e - $s + 1
        $bin = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
        $got = if (Test-Path $bin) { [long](Get-Item $bin).Length } else { [long]0 }
        if ($got -ge $want) { continue }
        $from = $s + $got
        $rest = Join-Path $partDir ("part_{0:D3}.rest" -f $i)
        $procs += Start-Process -FilePath 'curl.exe' -ArgumentList @(
            '-s', '-r', "$from-$e", '-o', $rest, '--max-time', '600', $Url
        ) -PassThru -NoNewWindow
    }
    Note ("  fresh connections: {0}" -f $procs.Count)
    Start-Sleep -Seconds $RoundSeconds

    # --- recycle connections (this is the whole point) ---
    foreach ($p in $procs) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 400
}

# --- final merge + concat ---
for ($i = 0; $i -lt $Parts; $i++) { Merge-RestInto $i }

$missing = 0
for ($i = 0; $i -lt $Parts; $i++) {
    $r = Get-PartRange $i
    $want = $r[1] - $r[0] + 1
    $bin = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
    $got = if (Test-Path $bin) { [long](Get-Item $bin).Length } else { [long]0 }
    if ($got -ne $want) { Note ("  INCOMPLETE part {0:D3}: {1}/{2}" -f $i, $got, $want); $missing++ }
}
if ($missing -gt 0) { Note ("FAIL {0}: {1} incomplete part(s) - re-run to resume" -f $Out, $missing); exit 2 }

$target = Join-Path $raw $Out
if (Test-Path $target) { Remove-Item $target -Force }
$fs = [IO.File]::Create($target)
for ($i = 0; $i -lt $Parts; $i++) {
    $bin = Join-Path $partDir ("part_{0:D3}.bin" -f $i)
    $src = [IO.File]::OpenRead($bin)
    $src.CopyTo($fs, 4194304)
    $src.Close()
}
$fs.Close()

$final = (Get-Item $target).Length
Note ("DONE {0} = {1:N0} bytes (expect {2:N0})" -f $Out, $final, $total)
if ($final -eq $total) { Remove-Item "$partDir\*" -Force -ErrorAction SilentlyContinue }
