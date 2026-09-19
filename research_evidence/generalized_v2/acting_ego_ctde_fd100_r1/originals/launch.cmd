@echo off
REM ============================================================================
REM GENERALIZED-V2 acting-ego CTDE FD100 DEVELOPMENT diagnostic (100 updates)
REM Measured code SHA 68055e39768d5fa601e5960a9f08823b9e65c08f (branch task/ctde-acting-ego-conditioning, draft PR #75, clean)
REM DEVELOPMENT profile only. Identical to the FD100 comparator launch.cmd except:
REM measured code, --iterations 100, and the run directory.
REM ============================================================================
set "RUN_DIR=C:\gruns\graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3"
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
