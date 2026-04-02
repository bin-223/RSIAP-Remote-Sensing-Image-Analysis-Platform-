@echo off
setlocal
cd /d "%~dp0"

set "APP_DIR=%~dp0"
set "APP=unified_frontend.py"

set "RUNTIME_PRIMARY=%~dp0runtime\python"
set "RUNTIME_FALLBACK=%~dp0runtime\python"

set "HOST=127.0.0.1"
set "PORT=8501"
set "CHECK_ONLY=0"
set "OPEN_BROWSER=1"

:parse_args
if "%~1"=="" goto after_args
if /i "%~1"=="--check" set "CHECK_ONLY=1"
if /i "%~1"=="--no-browser" set "OPEN_BROWSER=0"
shift
goto parse_args

:after_args

if not exist "%APP%" (
  echo unified_frontend.py not found in %APP_DIR%
  pause > nul
  exit /b 1
)

if exist "%RUNTIME_PRIMARY%\python.exe" (
  set "PY=%RUNTIME_PRIMARY%\python.exe"
  set "RUNTIME_ROOT=%RUNTIME_PRIMARY%"
  set "RUNTIME_NAME=unified_runtime"
) else if exist "%RUNTIME_FALLBACK%\python.exe" (
  set "PY=%RUNTIME_FALLBACK%\python.exe"
  set "RUNTIME_ROOT=%RUNTIME_FALLBACK%"
  set "RUNTIME_NAME=unified_runtime"
) else (
  echo No bundled Python runtime was found.
  echo Keep this folder when moving to another PC:
  echo   runtime\python
  pause > nul
  exit /b 1
)

set "PATH=%RUNTIME_ROOT%;%RUNTIME_ROOT%\Scripts;%RUNTIME_ROOT%\Library\bin;%RUNTIME_ROOT%\DLLs;%PATH%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

for /f %%P in ('powershell -NoProfile -Command "$ports = @{}; foreach ($listener in [System.Net.NetworkInformation.IPGlobalProperties]::GetActiveTcpListeners()) { $ports[$listener.Port] = $true }; foreach ($candidate in 8501..8510) { if (-not $ports.ContainsKey($candidate)) { Write-Output $candidate; break } }"') do (
  set "PORT=%%P"
)

if not defined PORT (
  echo No free port found in range 8501-8510.
  pause > nul
  exit /b 1
)

echo Using runtime: %RUNTIME_NAME%
if "%CHECK_ONLY%"=="1" (
  echo Running launcher self-check...
  "%PY%" -c "import sys; sys.path.insert(0, r'segmentation_pytorch'); import streamlit, torch, cv2, onnxruntime, network; print('self_check_ok'); print(sys.executable)"
  exit /b %errorlevel%
)

echo Launching unified frontend...
echo URL: http://%HOST%:%PORT%
if "%OPEN_BROWSER%"=="1" (
  echo Browser auto-open: on
  start "" powershell -NoProfile -WindowStyle Hidden -Command "$hostIp='%HOST%'; $port=%PORT%; for ($i=0; $i -lt 120; $i++) { try { if (Test-NetConnection -ComputerName $hostIp -Port $port -InformationLevel Quiet) { Start-Process ('http://{0}:{1}' -f $hostIp, $port); break } } catch {} ; Start-Sleep -Milliseconds 500 }"
) else (
  echo Browser auto-open: off
)
echo.

"%PY%" -m streamlit run "%APP%" --server.address %HOST% --server.port %PORT% --server.headless true
if errorlevel 1 (
  echo.
  echo Launch failed.
  pause > nul
  exit /b 1
)

endlocal
