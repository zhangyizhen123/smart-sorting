@echo off
setlocal
cd /d "%~dp0"
set "SORTING_FOCUS_PYTHON=%~dp0.venv-vision\Scripts\python.exe"
set "SORTING_FOCUS_PIPELINE=%~dp0models\trained\stl_shape_focus_v2\pipeline.json"
if not exist "%SORTING_FOCUS_PIPELINE%" (
  echo Focused model evaluation is not finished yet. Use the original demo for now.
  pause
  exit /b 1
)
"%SORTING_FOCUS_PYTHON%" "%~dp0vision\scripts\serve_stl_demo.py" --pipeline "%SORTING_FOCUS_PIPELINE%" %*
if errorlevel 1 pause
