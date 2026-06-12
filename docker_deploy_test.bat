@echo off
REM ============================================================
REM LocalOCR — Docker redeploy + smoke test (v0.5.0)
REM Builds the CPU image, starts it, waits for health,
REM then runs the end-to-end OCR test INSIDE the container.
REM ============================================================
echo [1/4] Stopping old container...
docker compose down localocr 2>nul

echo [2/4] Building image (first build downloads models, can take 10-20 min)...
docker compose build localocr || goto :fail

echo [3/4] Starting container...
docker compose up -d localocr || goto :fail

echo Waiting for web app to become healthy (http://localhost:7870)...
set RETRIES=0
:wait
timeout /t 10 /nobreak >nul
curl -sf http://localhost:7870/ >nul 2>&1 && goto :healthy
set /a RETRIES+=1
if %RETRIES% lss 30 goto :wait
echo App did not come up - showing logs:
docker compose logs --tail 50 localocr
goto :fail

:healthy
echo Web app is UP: http://localhost:7870
echo [4/4] Running end-to-end OCR test inside the container...
docker compose exec localocr python run_e2e_test.py || goto :fail
echo.
echo ALL GOOD - open http://localhost:7870 and convert your PDF.
exit /b 0

:fail
echo DEPLOY/TEST FAILED - see output above.
exit /b 1
