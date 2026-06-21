@echo off
setlocal
cd /d "%~dp0"

set "RSCRIPT=Rscript"
where %RSCRIPT% >nul 2>nul
if %ERRORLEVEL% neq 0 (
  for /f "tokens=2,*" %%A in ('reg query "HKLM\SOFTWARE\R-core\R" /v InstallPath 2^>nul ^| find "InstallPath"') do (
    if exist "%%B\bin\Rscript.exe" (
      set "RSCRIPT=%%B\bin\Rscript.exe"
      goto found_rscript
    )
    if exist "%%B\bin\x64\Rscript.exe" (
      set "RSCRIPT=%%B\bin\x64\Rscript.exe"
      goto found_rscript
    )
  )
  for /f "delims=" %%I in ('dir /b /ad /o-n "C:\Program Files\R\R-*" 2^>nul') do (
    if exist "C:\Program Files\R\%%I\bin\Rscript.exe" (
      set "RSCRIPT=C:\Program Files\R\%%I\bin\Rscript.exe"
      goto found_rscript
    )
    if exist "C:\Program Files\R\%%I\bin\x64\Rscript.exe" (
      set "RSCRIPT=C:\Program Files\R\%%I\bin\x64\Rscript.exe"
      goto found_rscript
    )
  )
  for /f "delims=" %%I in ('dir /b /ad /o-n "F:\Program Files\R\R-*" 2^>nul') do (
    if exist "F:\Program Files\R\%%I\bin\Rscript.exe" (
      set "RSCRIPT=F:\Program Files\R\%%I\bin\Rscript.exe"
      goto found_rscript
    )
    if exist "F:\Program Files\R\%%I\bin\x64\Rscript.exe" (
      set "RSCRIPT=F:\Program Files\R\%%I\bin\x64\Rscript.exe"
      goto found_rscript
    )
  )
)

:found_rscript
if /I "%RSCRIPT%"=="Rscript" (
  where %RSCRIPT% >nul 2>nul
  if %ERRORLEVEL% neq 0 (
    echo Rscript was not found. Install R and ensure Rscript is on PATH, or update install_dependencies.bat with your local R path.
    pause
    exit /b 1
  )
) else (
  if not exist "%RSCRIPT%" (
  echo Rscript was not found. Install R and ensure Rscript is on PATH, or update install_dependencies.bat with your local R path.
  pause
  exit /b 1
  )
)

"%RSCRIPT%" "scripts\install_packages.R"
if %ERRORLEVEL% neq 0 pause
endlocal
