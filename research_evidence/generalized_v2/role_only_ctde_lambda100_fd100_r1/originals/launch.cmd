@echo off
REM ============================================================================
REM GENERALIZED-V2 role-only acting-ego CTDE, gae_lambda 1.0, FD100 DEVELOPMENT diagnostic
REM Measured code SHA 68055e39768d5fa601e5960a9f08823b9e65c08f (isolated detached worktree C:\grolelambda1, clean)
REM Config: run-local preset = primary comparator run_config.json:/train_config with ONLY
REM /ctde/gae_lambda 0.95 -> 1.0 and /output_dir changed. DEVELOPMENT profile only.
REM ============================================================================
set "RUN_DIR=C:\gruns\graph_rl_v2_role_only_ctde_lambda100_fd100_r1_seed3000000_68055e3"
set "REPO=C:\grolelambda1"
set "PYTHONPATH=src"

cd /d "%REPO%"
>"%RUN_DIR%\invocation_start_local.txt" echo start_local=%DATE% %TIME%

call conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train ^
 --config "%RUN_DIR%\train_config_preset.json" ^
 > "%RUN_DIR%\training_console.log" 2>&1

set RC=%ERRORLEVEL%
>"%RUN_DIR%\native_exit_code.txt" echo %RC%
>>"%RUN_DIR%\invocation_start_local.txt" echo end_local=%DATE% %TIME%
exit /b %RC%
