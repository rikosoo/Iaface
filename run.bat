@echo off
REM Sobe o servidor local do Iaface em http://localhost:8000
REM Basta dar dois cliques neste arquivo, ou rodar "run.bat" no terminal.

setlocal
cd /d "%~dp0"

REM --- 1. Achar um Python compativel -------------------------------------
REM O facenet-pytorch exige torch <2.3, que so tem instalador ate o 3.12.
REM O py.exe (Python Launcher) deixa escolher a versao mesmo com outra no PATH.

set PY=
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set PY=py -%%V
  )
)

if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>nul
  if not errorlevel 1 set PY=python
)

if not defined PY (
  echo.
  echo ============================================================
  echo   Nenhum Python compativel encontrado.
  echo.
  echo   Este projeto precisa do Python 3.10, 3.11 ou 3.12.
  echo   As versoes 3.13 e 3.14 ainda nao tem PyTorch compativel.
  echo.
  echo   Instale o Python 3.12 em:
  echo   https://www.python.org/downloads/release/python-31210/
  echo   ^(marque "Add python.exe to PATH" na primeira tela^)
  echo.
  echo   Voce pode manter o Python que ja tem instalado: os dois
  echo   convivem sem problema.
  echo ============================================================
  goto :erro
)

echo Usando: %PY%
%PY% --version

REM --- 2. Ambiente virtual ------------------------------------------------
REM Testamos o python.exe de dentro do .venv, e nao so a pasta: uma instalacao
REM interrompida deixa a pasta criada mas vazia, e ai o erro so aparecia la na
REM frente como "No module named numpy".

if not exist .venv\Scripts\python.exe (
  if exist .venv (
    echo Ambiente virtual incompleto - refazendo do zero...
    rmdir /s /q .venv
  )
  echo Criando o ambiente virtual...
  %PY% -m venv .venv
  if errorlevel 1 goto :erro
)

REM --- 3. Dependencias ----------------------------------------------------
REM O marcador evita reinstalar a cada execucao, mas some se o requirements
REM mudar de conteudo, forcando a atualizacao.

if not exist .venv\Scripts\uvicorn.exe goto :instalar
if not exist .venv\.deps-ok goto :instalar
goto :base

:instalar
echo.
echo Instalando as dependencias. Sao cerca de 2 GB - va tomar um cafe.
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo A instalacao falhou. Se a mensagem acima fala em "No matching
  echo distribution found for torch", o Python usado nao e compativel:
  echo instale o 3.12 e rode este arquivo de novo.
  goto :erro
)
echo ok > .venv\.deps-ok

:base
if not exist data\actors.npz (
  echo.
  echo Base de atores ausente - montando agora. Leva alguns minutos.
  .venv\Scripts\python -m scripts.build_actors
  if errorlevel 1 goto :erro
)

REM --- 4. Servidor --------------------------------------------------------
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
