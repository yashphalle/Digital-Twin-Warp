Write-Host "STARTING CAMERA GROUP 1 (Cameras 1-6)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
$env:CAMERA_GROUP = "GROUP1"
python gpu_11camera_configurable.py
