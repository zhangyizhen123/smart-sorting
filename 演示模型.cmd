@echo off
setlocal
set "SORTING_DEMO_PYTHON=%~dp0.venv-vision\Scripts\python.exe"
if not exist "%SORTING_DEMO_PYTHON%" (
  echo The vision Python environment is missing.
  echo Open docs\stl_shape_v1_demo\index.html to view the saved demonstration.
  echo See vision\STL*.md for environment setup.
  pause
  exit /b 1
)
"%SORTING_DEMO_PYTHON%" "%~dp0vision\scripts\predict_stl_demo.py" %*
if errorlevel 1 (
  echo Demo inference failed. The saved demo is still available in docs\stl_shape_v1_demo\index.html.
  pause
  exit /b 1
)
endlocal
