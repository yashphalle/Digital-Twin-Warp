Write-Host "STARTING CAMERA GROUP 2 (Cameras 7-11)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
$env:CAMERA_GROUP = "GROUP2"
python gpu_11camera_configurable.py
