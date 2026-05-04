@echo off

echo Checking if container exists...

docker ps -a --format "{{.Names}}" | findstr housing-api-container >nul

IF %ERRORLEVEL%==0 (
    echo Container exists. Starting it...
    docker start -a housing-api-container
) ELSE (
    echo Container does not exist. Creating it...
    docker run -p 8000:8000 --name housing-api-container housing-api
)

pause