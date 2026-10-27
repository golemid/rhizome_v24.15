<#
    trackb_bitexact_gate.ps1 -- RHIZOME v24.02 (DRAFT) Track B bit-exactness gate
    v24.02 P1.2 deliverable (c): the machine gate that converts P1.2 from an
    aspiration into a measurable property. Verdicts come from SHA-256 hashes
    only (timing-free by design; no wall-clock value participates in any
    verdict).

    UNVERIFIED UNTIL TRACK B: this script is DESIGNED to run on the user's
    Windows/CUDA box; its output IS the Track B measurement.

    MEASUREMENT DESIGN
      N independent `--verify` runs of rhizome_app. Before every run the three
      Test-7 export artifacts are deleted, so each hash is provably this run's
      export (a stale file can never masquerade as an export). After every run:
        - the suite banner must read "Suite Summary: 20 Passed, 0 Failed"
          (a red or missing banner invalidates the gate), and
        - SHA-256 is taken of each artifact.
      Per artifact, across all N runs:
        - 1 distinct hash                     -> BIT-EXACT over N runs
        - M > 1 distinct hashes               -> NOT bit-exact; M and the
          run indices per distinct hash are reported.

    EXIT CODES
      0 = measured bit-exact (all artifacts identical across all runs)
      1 = measured NOT bit-exact (a reported measurement, not a build failure)
      2 = gate invalid (exe missing / banner not 20/20 / artifact not produced)

    USAGE (from native\, standard build tree; same CWD convention as run.bat):
      powershell -NoProfile -ExecutionPolicy Bypass -File tools\trackb_bitexact_gate.ps1
      optional: -Runs 10 -BuildDir build\Release -ExeName rhizome_app.exe
#>
param(
    [int]$Runs = 10,
    [string]$BuildDir = "build\Release",
    [string]$ExeName = "rhizome_app.exe"
)

$ErrorActionPreference = "Stop"

# Work from the script's grandparent directory (native\), matching run.bat's
# convention: exe at build\Release\rhizome_app.exe, assets\ beside the CWD.
$nativeDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $nativeDir

$exePath = Join-Path $BuildDir $ExeName
if (-not (Test-Path $exePath)) {
    Write-Host "GATE INVALID: exe not found: $exePath"
    exit 2
}

$artifacts = @(
    "assets\root_clean.obj",
    "assets\root_dirty_inverted.obj",
    "assets\root_filleted_hires.obj"
)

$logDir = "bitexact_gate_logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# Per-artifact ordered per-run hashes
$hashes = @{}
foreach ($a in $artifacts) { $hashes[$a] = @() }

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host ("[run {0}/{1}] launching --verify ..." -f $i, $Runs)

    # Prove artifacts fresh: delete before the run so every hash is this run's export
    foreach ($a in $artifacts) {
        if (Test-Path $a) { Remove-Item $a -Force }
    }

    $log = Join-Path $logDir ("run_{0}.txt" -f $i)
    & $exePath --verify 2>&1 | Out-File -FilePath $log -Encoding utf8

    $txt = Get-Content $log -Raw
    if (-not $txt) {
        Write-Host ("GATE INVALID: run {0} produced no output; see {1}" -f $i, $log)
        exit 2
    }
    if ($txt -notmatch "Suite Summary: 20 Passed, 0 Failed") {
        Write-Host ("GATE INVALID: run {0} banner is not 20/20; see {1}" -f $i, $log)
        exit 2
    }

    foreach ($a in $artifacts) {
        if (-not (Test-Path $a)) {
            Write-Host ("GATE INVALID: run {0} did not produce {1}" -f $i, $a)
            exit 2
        }
        $h = (Get-FileHash -Algorithm SHA256 $a).Hash
        $hashes[$a] += $h
        Write-Host ("    {0}  {1}" -f $a, $h)
    }
}

# Verdict -- hashes only
$summary = @()
$allBitExact = $true
foreach ($a in $artifacts) {
    $distinct = @($hashes[$a] | Select-Object -Unique)
    if ($distinct.Count -eq 1) {
        $summary += ("{0}: BIT-EXACT across {1} runs (SHA-256 {2})" -f $a, $Runs, $distinct[0])
    } else {
        $allBitExact = $false
        $summary += ("{0}: NOT bit-exact -- {1} distinct hashes across {2} runs:" -f $a, $distinct.Count, $Runs)
        foreach ($d in $distinct) {
            $runsAt = @()
            for ($i = 0; $i -lt $Runs; $i++) {
                if ($hashes[$a][$i] -eq $d) { $runsAt += ($i + 1) }
            }
            $summary += ("    {0}  runs: {1}" -f $d, ($runsAt -join ", "))
        }
    }
}

if ($allBitExact) {
    $verdict = "BIT-EXACT (all artifacts identical across {0} runs)" -f $Runs
} else {
    $verdict = "NOT bit-exact (per-artifact breakdown follows)"
}

$lines = @()
$lines += "RHIZOME v24.02 Track B bit-exactness gate (P1.2 deliverable c)"
$lines += ("Runs: {0}   Exe: {1}   Date: {2}" -f $Runs, (Join-Path $BuildDir $ExeName), (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$lines += ("Verdict: " + $verdict)
$lines += ""
$lines += $summary
$lines | Out-File -FilePath "bitexact_gate_summary.txt" -Encoding utf8

Write-Host ""
$lines | ForEach-Object { Write-Host $_ }
Write-Host ""
if ($allBitExact) {
    Write-Host "RESULT: PASS -- measured bit-exact"
    exit 0
} else {
    Write-Host "RESULT: MEASURED NOT BIT-EXACT -- a measurement, not a build failure"
    exit 1
}
