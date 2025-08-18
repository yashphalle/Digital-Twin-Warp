@echo off
echo ========================================
echo Redis Command Line Interface (CLI)
echo ========================================

echo.
echo Basic Redis Commands:
echo ---------------------
echo 1. Test connection:
echo    Redis\redis-cli.exe ping
echo.

echo 2. Show all keys:
echo    Redis\redis-cli.exe keys "*"
echo.

echo 3. Show track keys only:
echo    Redis\redis-cli.exe keys "camera_*:track_*"
echo.

echo 4. Count total keys:
echo    Redis\redis-cli.exe dbsize
echo.

echo 5. Show Redis info:
echo    Redis\redis-cli.exe info
echo.

echo 6. Monitor Redis activity:
echo    Redis\redis-cli.exe monitor
echo.

echo 7. Get specific track data:
echo    Redis\redis-cli.exe hgetall "camera_8:track_8001"
echo.

echo 8. Clear all data (DANGER!):
echo    Redis\redis-cli.exe flushall
echo.

echo ========================================
echo Quick Tests:
echo ========================================

echo.
echo Testing Redis connection...
Redis\redis-cli.exe ping

echo.
echo Showing total keys in Redis...
Redis\redis-cli.exe dbsize

echo.
echo Showing all track keys...
Redis\redis-cli.exe keys "camera_*:track_*"

echo.
echo ========================================
echo Interactive Redis CLI:
echo ========================================
echo Starting interactive Redis CLI...
echo Type 'exit' to quit the Redis CLI
echo.

Redis\redis-cli.exe
