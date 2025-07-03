@echo off
echo 🔥 STARTING CAMERA GROUP 1 (Cameras 1-6)
echo ==========================================
echo.
echo OPTIONS:
echo 1. Run Group 1 (Cameras 1-6)
echo 2. Run Single Camera 8 (for testing)
echo 3. Run Coordinate Verification
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo Running Camera Group 1...
    set CAMERA_GROUP=GROUP1
    python gpu_11camera_configurable.py
) else if "%choice%"=="2" (
    echo Running Single Camera 8...
    set CAMERA_GROUP=SINGLE8
    python gpu_11camera_configurable.py
) else if "%choice%"=="3" (
    echo Running Coordinate Verification...
    python verify_coordinates.py 8
) else (
    echo Invalid choice. Running Group 1 by default...
    set CAMERA_GROUP=GROUP1
    python gpu_11camera_configurable.py
)
pause
