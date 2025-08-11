@echo off
echo ========================================
echo Redis Installation Script for Windows
echo ========================================

echo.
echo Installing Redis Python client...
pip install redis

echo.
echo ========================================
echo Redis Server Installation Options:
echo ========================================

echo.
echo Option 1: Docker (Recommended)
echo ------------------------------
echo If you have Docker installed, run:
echo docker run -d -p 6379:6379 --name redis redis:latest
echo.

echo Option 2: Windows Binary
echo ------------------------
echo 1. Download Redis from: https://github.com/microsoftarchive/redis/releases
echo 2. Extract to C:\Redis
echo 3. Run: C:\Redis\redis-server.exe
echo.

echo Option 3: WSL (Windows Subsystem for Linux)
echo -------------------------------------------
echo 1. Install WSL: wsl --install
echo 2. Install Redis: sudo apt update && sudo apt install redis-server
echo 3. Start Redis: redis-server
echo.

echo Option 4: Chocolatey
echo -------------------
echo If you have Chocolatey installed:
echo choco install redis-64 -y
echo.

echo ========================================
echo Testing Redis Installation
echo ========================================
echo.
echo Running Redis test script...
python test_redis_setup.py

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo If Redis test passed, you can now run:
echo python test_pipline_with_BoT_persistence.py
echo.
pause
