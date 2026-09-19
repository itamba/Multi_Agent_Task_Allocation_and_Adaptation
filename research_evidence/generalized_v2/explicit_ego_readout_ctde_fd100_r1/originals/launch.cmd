@echo off
REM ============================================================================
REM GENERALIZED-V2 explicit acting-ego readout CTDE FD100 DEVELOPMENT diagnostic (100 updates)
REM Measured code SHA 1a1e0c953c54e9d3f46c871158d5ab6bdd881f24 (branch task/ctde-acting-ego-conditioning, draft PR #75, clean)
REM DEVELOPMENT profile only. Identical to the role-only acting-ego launch.cmd except:
REM measured code (explicit critic readout) and the run directory.
REM ============================================================================
set "RUN_DIR=C:\gruns\graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9"
set "REPO=C:\Users\Itama\PycharmProjects\Multi_Agent_Task_Allocation_and_Adaptation"
set "MANIFEST=C:\gra\benchmarks\v2_preflight_seed2000000_ae42cb0\benchmark_manifest.json"
set "PYTHONPATH=src"

cd /d "%REPO%"
>"%RUN_DIR%\invocation_start_local.txt" echo start_local=%DATE% %TIME%

call conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train ^
 --iterations 100 ^
 --episodes 8 ^
 --seed 3000000 ^
 --out "%RUN_DIR%" ^
 --checkpoint-every 25 ^
 --eval-every 25 ^
 --eval-episodes 8 ^
 --training-mode ctde ^
 --episode-design generalized_v2 ^
 --match-aou-backend p1_milp_v1 ^
 --fuel-damage-mode seeded_variable ^
 --fuel-damage-probability 1.0 ^
 --generalized-max-attempts-per-iteration 12 ^
 --benchmark-manifest "%MANIFEST%" ^
 --benchmark-profile development ^
 --actor-gradient-diagnostics ^
 > "%RUN_DIR%\training_console.log" 2>&1

set RC=%ERRORLEVEL%
>"%RUN_DIR%\native_exit_code.txt" echo %RC%
>>"%RUN_DIR%\invocation_start_local.txt" echo end_local=%DATE% %TIME%
exit /b %RC%
