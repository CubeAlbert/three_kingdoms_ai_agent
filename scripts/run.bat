@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ===========================================================================
REM San Guo AI Agent - Windows Launcher
REM ===========================================================================

REM ---- LLM Chat Provider (required) ----
REM DeepSeek:
REM   set LLM_BASE_URL=https://api.deepseek.com/v1
REM   set LLM_AUTH_ENABLED=true
REM   set LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
REM   set LLM_MODEL=deepseek-chat
REM
REM Ollama (local):
REM   set LLM_BASE_URL=http://localhost:11434/v1
REM   set LLM_AUTH_ENABLED=false
REM   set LLM_MODEL=qwen2.5:7b

if "%LLM_BASE_URL%"=="" set LLM_BASE_URL=https://api.deepseek.com/v1
if "%LLM_AUTH_ENABLED%"=="" set LLM_AUTH_ENABLED=true
if "%LLM_MODEL%"=="" set LLM_MODEL=deepseek-chat

REM ---- Debug mode (optional) ----
REM   set DEBUG=true   -> show RAG hits/misses, agent switches, init details
REM   set DEBUG=false  -> quiet mode (default)

REM ---- Embedding Provider (optional, falls back to LLM provider) ----
REM DeepSeek has no embedding API; use a separate provider.
REM Example (Volcengine):
REM   set EMBED_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
REM   set EMBED_AUTH_ENABLED=true
REM   set EMBED_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
REM   set EMBED_MODEL=text-embedding-v4

REM ---- Launch ----
cd /d "%~dp0\.."

echo.
echo [San Guo AI Agent] Starting...
echo   LLM  : %LLM_MODEL% @ %LLM_BASE_URL%
if not "%EMBED_MODEL%"=="" (
    echo   Embed: %EMBED_MODEL% @ %EMBED_BASE_URL%
) else (
    echo   Embed: ^(fallback to LLM provider^)
)
echo.

if exist ".venv\Scripts\python.exe" (
    echo Using venv: .venv
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

%PYTHON% -m three_kingdoms_ai_agent.main

endlocal
pause
