@echo off
chcp 65001 > nul
echo ======================================================
echo    Yapay Zeka Modeli Eğitimi Başlatılıyor...
echo ======================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python train.py
) else (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" train.py
)

pause
