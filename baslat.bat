@echo off
chcp 65001 > nul
echo ======================================================
echo    Yapay Zeka Web Arayüzü Başlatılıyor...
echo ======================================================
echo Tarayıcınızda açılacak adres: http://127.0.0.1:7860
echo.

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python app.py
) else (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" app.py
)

pause
