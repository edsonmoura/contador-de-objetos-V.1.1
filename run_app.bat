@echo off
cd /d "%~dp0"
python main.py
echo.
echo Se o app fechou com erro, veja tambem o arquivo app_error.log.
pause

