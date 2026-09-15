$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\Itama\PycharmProjects\Multi_Agent_Task_Allocation_and_Adaptation'
$Sweep = 'C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_sweep_ae42cb0'
$Sha = 'ae42cb01677f94868b2873008d87be677e31f0c8'
$Manifest = 'C:\Users\Itama\PycharmProjects\graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0\benchmark_manifest.json'
$ManifestSha = 'DD72AFC9CC0D2D1FE494DDBEBE53734DC36BD5890997125D3E96A2A59641A103'
$Log = Join-Path $Sweep 'sweep_log.txt'

function Say($m) { $l = "$((Get-Date).ToUniversalTime().ToString('o')) $m"; Add-Content -Path $Log -Value $l -Encoding UTF8 }
function ManHash { (Get-FileHash -Algorithm SHA256 $Manifest).Hash }
function GitHead { (& git -C $Repo rev-parse HEAD).Trim() }
function GitDirty { (& git -C $Repo status --porcelain) -join "`n" }

$env:PYTHONPATH = "$Repo\src"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
Set-Location $Repo

$arms = @(
  @{ name='smallbatch'; iters=750; eps=4;  maxatt=6;  every=50; fd='0.5' },
  @{ name='largebatch'; iters=150; eps=20; maxatt=30; every=10; fd='0.5' },
  @{ name='fd80';       iters=375; eps=8;  maxatt=12; every=25; fd='0.8' }
)

Say "SWEEP START"
foreach ($a in $arms) {
  $out = "C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_$($a.name)_seed3000000_ae42cb0"
  $console = "$out.official.console.log"
  $timing = "$out.official.timing.txt"
  $check = "$out.official.check.txt"
  $h0 = ManHash; $g0 = GitHead; $d0 = GitDirty
  if ($h0 -ne $ManifestSha -or $g0 -ne $Sha -or $d0 -ne '' -or (Test-Path $out) -or (Test-Path $console)) {
    Say "STOP pre-check failed arm=$($a.name) manifest=$h0 head=$g0 dirty=[$d0] out_exists=$(Test-Path $out) console_exists=$(Test-Path $console)"
    Set-Content -Path (Join-Path $Sweep 'SWEEP_DONE') -Value "STOPPED_PRECHECK $($a.name)"; exit 1
  }
  $cmdLine = "call conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --iterations $($a.iters) --episodes $($a.eps) --seed 3000000 --out `"$out`" --checkpoint-every $($a.every) --eval-every $($a.every) --eval-episodes 8 --training-mode ctde --episode-design generalized_v2 --match-aou-backend p1_milp_v1 --fuel-damage-mode seeded_variable --fuel-damage-probability $($a.fd) --fuel-damage-mild-probability 0.5 --generalized-max-attempts-per-iteration $($a.maxatt) --benchmark-manifest `"$Manifest`" --benchmark-profile development > `"$console`" 2>&1"
  $cmdFile = Join-Path $Sweep "arm_$($a.name).cmd"
  Set-Content -Path $cmdFile -Encoding ASCII -Value @("@echo off", "cd /d `"$Repo`"", $cmdLine, "exit /b %ERRORLEVEL%")
  Set-Content -Path $timing -Encoding ASCII -Value @("MANIFEST_SHA256_BEFORE=$h0", "GIT_HEAD_BEFORE=$g0", "GIT_DIRTY_BEFORE=$([int]($d0 -ne ''))", "RUN_START_UTC=$((Get-Date).ToUniversalTime().ToString('o'))")
  Say "ARM START $($a.name)"
  & cmd.exe /c "`"$cmdFile`""
  $rc = $LASTEXITCODE
  $h1 = ManHash; $g1 = GitHead; $d1 = GitDirty
  Add-Content -Path $timing -Encoding ASCII -Value @("RUN_END_UTC=$((Get-Date).ToUniversalTime().ToString('o'))", "EXIT_CODE=$rc", "MANIFEST_SHA256_AFTER=$h1", "GIT_HEAD_AFTER=$g1", "GIT_DIRTY_AFTER=$([int]($d1 -ne ''))")
  Say "ARM END $($a.name) exit=$rc manifest=$h1 head=$g1 dirty=[$d1]"
  $crash = Select-String -Path $console -Pattern 'Traceback|CRASH' -SimpleMatch:$false -Quiet
  if ($rc -ne 0 -or $h1 -ne $ManifestSha -or $g1 -ne $Sha -or $d1 -ne '' -or -not (Test-Path "$out\run_summary.json")) {
    Say "STOP post-run failure arm=$($a.name) traceback_in_log=$crash"
    Set-Content -Path (Join-Path $Sweep 'SWEEP_DONE') -Value "STOPPED_RUN $($a.name)"; exit 1
  }
  $checkOut = & cmd.exe /c "call conda run -n nlp_env --no-capture-output python `"$Sweep\check_arm.py`" `"$out`" $($a.iters) $($a.eps) $($a.maxatt) $($a.every) $($a.fd) 2>&1"
  $crc = $LASTEXITCODE
  Set-Content -Path $check -Encoding UTF8 -Value ($checkOut -join "`n")
  Say "CHECK $($a.name) exit=$crc traceback_in_log=$crash"
  if ($crc -ne 0) {
    Say "STOP accounting check failed arm=$($a.name)"
    Set-Content -Path (Join-Path $Sweep 'SWEEP_DONE') -Value "STOPPED_CHECK $($a.name)"; exit 1
  }
}
Say "SWEEP COMPLETE"
Set-Content -Path (Join-Path $Sweep 'SWEEP_DONE') -Value "COMPLETE"
