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
"%SORTING_DEMO_PYTHON%" "%~dp0vision\scripts\serve_stl_demo.py" %*
if errorlevel 1 (
  echo Demo server stopped with an error. The saved demo is still available in docs\stl_shape_v1_demo\index.html.
  pause
  exit /b 1
)
endlocal
