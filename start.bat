@echo off
chcp 65001 >nul
setlocal
title PlateVision - Nhan dang bien so xe may
cd /d "%~dp0"

set "URL=http://127.0.0.1:5000"
set "PY=.venv\Scripts\python.exe"

echo ================================================
echo   PlateVision - Nhan dang bien so xe may
echo ================================================
echo.

rem --- Neu server da chay san thi chi mo trinh duyet ---
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%/api/status' | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo Server dang chay san. Mo trinh duyet...
    start "" "%URL%"
    timeout /t 2 >nul
    exit /b 0
)

rem --- Lan dau: tao moi truong ao va cai thu vien ---
if not exist "%PY%" (
    echo [1/2] Chua co moi truong .venv - dang tao...
    where python >nul 2>nul
    if errorlevel 1 (
        echo LOI: Khong tim thay Python. Hay cai Python 3.10+ tu https://www.python.org/downloads/
        echo      va tick "Add Python to PATH" khi cai.
        pause
        exit /b 1
    )
    python -m venv .venv || goto :fail
    echo [2/2] Dang cai thu vien ^(lan dau mat vai phut^)...
    "%PY%" -m pip install --upgrade pip -q
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q || goto :fail
    "%PY%" -m pip install -r requirements.txt -q || goto :fail
    echo Cai dat xong.
    echo.
)

rem --- Mo trinh duyet khi server san sang (chay nen) ---
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%/api/status' | Out-Null; Start-Process '%URL%'; break } catch { Start-Sleep -Milliseconds 700 } }"

echo Dang khoi dong server tai %URL%
echo Trinh duyet se tu mo. Dong cua so nay de tat server.
echo.
"%PY%" -W ignore app.py
echo.
echo Server da dung.
pause
exit /b 0

:fail
echo.
echo LOI: Cai dat that bai. Kiem tra ket noi mang roi chay lai start.bat.
pause
exit /b 1
