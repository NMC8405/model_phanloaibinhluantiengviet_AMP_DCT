@echo off
chcp 65001 >nul
REM run_train.bat
REM ==============
REM Chay toan bo pipeline: chuan bi du lieu -> train -> danh gia tren tap test.
REM Dung tren Windows (vd may co GPU RTX 4060 cua nhom).
REM
REM Cach dung:
REM   scripts\run_train.bat
REM
REM Truoc khi chay: dam bao da (1) cai dat requirements.txt, (2) co file
REM data\raw\comments.csv dung schema (xem README.md).

cd /d "%~dp0\.."

echo ============================================
echo BUOC 1/4: Chuan bi du lieu (prepare_data.py)
echo ============================================
python -m scripts.prepare_data
if errorlevel 1 goto :error

echo.
echo ============================================
echo BUOC 2/4: Huan luyen mo hinh (train.py)
echo ============================================
python -m src.train
if errorlevel 1 goto :error

echo.
echo ============================================
echo BUOC 3/4: Danh gia tren tap test (evaluate.py)
echo ============================================
python -m src.evaluate
if errorlevel 1 goto :error

echo.
echo ============================================
echo BUOC 4/4: Tao bao cao tong hop (generate_report.py)
echo ============================================
python -m scripts.generate_report
if errorlevel 1 goto :error

echo.
echo HOAN TAT! Xem ket qua tai outputs\reports\
pause
exit /b 0

:error
echo.
echo CO LOI XAY RA. Xem log ben tren de biet chi tiet.
pause
exit /b 1
