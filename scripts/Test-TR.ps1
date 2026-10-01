# Cardaman TR - self-test for the beverage pilots. Run from anywhere in PowerShell:
#   powershell -ExecutionPolicy Bypass -File C:\dev\Cardaman\scripts\Test-TR.ps1
# Writes everything into C:\dev\Cardaman\evaluation\selftest\<timestamp>\ and prints SUMMARY.txt at the end;
# that folder (or just SUMMARY.txt) is what to share. Steps 1-4 need no model; step 5 needs Ollama with the pinned
# models and the GPU, and runs only with -WithModel. -SkipFull leaves out the full-suite regression (step 2).
param([switch]$WithModel, [switch]$SkipFull)
$ErrorActionPreference = 'Continue'
$root = 'C:\dev\Cardaman'
$py = "$root\.venv\Scripts\python.exe"
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "$root\evaluation\selftest\$stamp"
New-Item -ItemType Directory -Force $out | Out-Null
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'; $env:PYTHONPATH = "$root\backend\src"
Set-Location "$root\backend"
$summary = @()
$summary += "Cardaman TR self-test $stamp  commit $(git -C $root rev-parse --short HEAD)  branch $(git -C $root rev-parse --abbrev-ref HEAD)"

function Run-Capture([string]$file, [string[]]$arguments) {
    # python's stdout and stderr into files (no pipe buffers: a full suite writes far more than a pipe holds), joined as UTF-8
    $quoted = $arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $proc = Start-Process -FilePath $py -ArgumentList $quoted -WorkingDirectory "$root\backend" -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$file.out" -RedirectStandardError "$file.err"
    $bytes = [System.IO.File]::ReadAllBytes("$file.out") + [System.IO.File]::ReadAllBytes("$file.err")
    [System.IO.File]::WriteAllBytes($file, $bytes)
    Remove-Item "$file.out", "$file.err" -ErrorAction SilentlyContinue
    return $proc.ExitCode
}
function Summarize([string]$kind, [string]$file) {
    return (& $py "scripts\tr_selftest_summary.py" $kind $file 2>&1) -join ' '
}

# 1. TR unit tests (rule reader, comparer, adjudication replay, engines) - about 2 minutes
Run-Capture "$out\1-tr-tests.txt" @('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_tr_*.py') | Out-Null
$summary += "1 TR tests: " + ((Get-Content "$out\1-tr-tests.txt" | Where-Object { $_ -match '^(Ran|OK|FAILED)' }) -join ' | ')

# 2. full suite against the recorded baseline (242 known artefact failures), by identity and reason - about 2 minutes
if (-not $SkipFull) {
    Run-Capture "$out\2-full-suite.txt" @('-m', 'unittest', 'discover', '-s', 'tests') | Out-Null
    Run-Capture "$out\2-regression.txt" @('scripts\regress_compare.py', "$root\evaluation\reports\regression-baseline\full-output.txt", "$out\2-full-suite.txt") | Out-Null
    $summary += "2 regression: " + ((Get-Content "$out\2-regression.txt" | Select-Object -Last 1))
}

# 3. the three pilots through every sector engine they fall under, replaying the recorded run (no model)
$rec = "$root\backend\src\regchain\tr\data\evaluation\recorded\20261001"
foreach ($profile in 'tr-bev-pilot-alcohol-integrated', 'tr-bev-pilot-non-alcohol-bottler', 'tr-bev-pilot-alcohol-import') {
    Run-Capture "$out\3-assess-$profile.summary.json" @('-m', 'regchain.tr', 'assess', '--profile', $profile, '--similarities', "$rec\similarities.json",
        '--adjudications', "$rec\adjudicate-dev.jsonl", '--adjudications', "$rec\adjudicate-holdout.jsonl", '--adjudications', "$rec\adjudicate-validation.jsonl",
        '--out', "$out\3-assess-$profile.json") | Out-Null
    $summary += "3 assess $profile -> " + (Summarize 'assess' "$out\3-assess-$profile.summary.json")
}

# 4. the development evaluation: every labelled task, DEV / HOLDOUT / VALIDATION, with the recorded run replayed
Run-Capture "$out\4-evaluate.json" @('-m', 'regchain.tr', 'ai', 'evaluate', '--similarities', "$rec\similarities.json",
    '--adjudications', "$rec\adjudicate-dev.jsonl", '--adjudications', "$rec\adjudicate-holdout.jsonl", '--adjudications', "$rec\adjudicate-validation.jsonl") | Out-Null
$summary += "4 evaluate: " + (Summarize 'evaluate' "$out\4-evaluate.json")

# 5. optional: a live leg with the local models (embeddings + strong-model adjudication) on one pilot - GPU, 10-60 minutes
if ($WithModel) {
    $env:CARDAMAN_MODE = 'development'; $env:OLLAMA_BASE_URL = 'http://localhost:11434'
    $env:LLM_PROVIDER = 'ollama'; $env:LLM_MODEL = 'qwen3:4b'; $env:LLM_THINKING = 'off'
    $env:JUDGE_MODEL = 'qwen3:8b'; $env:JUDGE_THINKING = 'on'; $env:JUDGE_NUM_CTX = '8192'; $env:EMBED_MODEL = 'bge-m3:latest'
    Run-Capture "$out\5-live-gaps.json" @('-m', 'regchain.tr', 'gaps', '--profile', 'tr-bev-pilot-alcohol-integrated',
        '--regulation', 'TR:KANUN:4250', '--regulation', 'TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM',
        '--selective', '--embed', '--adjudicate', '--similarities', "$out\5-similarities.json", '--adjudications', "$out\5-adjudications.jsonl") | Out-Null
    $summary += "5 live: " + (Summarize 'live' "$out\5-live-gaps.json")
}

[System.IO.File]::WriteAllLines("$out\SUMMARY.txt", $summary, (New-Object System.Text.UTF8Encoding $false))
Get-Content "$out\SUMMARY.txt"
Write-Host "`nfolder: $out"
