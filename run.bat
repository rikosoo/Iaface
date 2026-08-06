@echo off
REM Sobe o servidor local do Iaface em http://localhost:8000
REM Basta dar dois cliques neste arquivo, ou rodar "run.bat" no terminal.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python nao encontrado. Instale a versao 3.11 ou 3.12 em python.org
  echo e marque a opcao "Add python.exe to PATH" durante a instalacao.
  goto :erro
)

if not exist .venv (
  echo Criando o ambiente virtual...
  python -m venv .venv
  if errorlevel 1 goto :erro

  echo Instalando as dependencias ^(baixa cerca de 2 GB, demora^)...
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
  if errorlevel 1 goto :erro
)

if not exist data\actors.npz (
  echo.
  echo Base de atores ausente - montando agora. Leva alguns minutos.
  .venv\Scripts\python -m scripts.build_actors
  if errorlevel 1 goto :erro
)

if "%PORT%"=="" set PORT=8000

echo.
echo ==========================================================
echo   Abra no navegador:  http://localhost:%PORT%
echo   Para parar o servidor, pressione Ctrl+C nesta janela.
echo ==========================================================
echo.

.venv\Scripts\uvicorn app.server:app --host 127.0.0.1 --port %PORT%
goto :fim

:erro
echo.
echo Algo deu errado - a mensagem de erro esta logo acima.
pause
exit /b 1

:fim
pause
