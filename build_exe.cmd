@echo off
REM Script to build Python executable using PyInstaller

REM Set the main script and output executable name
set MAIN_SCRIPT=gui_main.py
set ICON_FILE=GUI/assets/icons/icons8-point-100.png
set OUTPUT_NAME=SmartLabellingTool
set BUILD_DIR=build
set DIST_DIR=dist
set SPEC_FILE=%OUTPUT_NAME%.spec

REM PyInstaller options
set CLEAN_OPTION=--clean
set ONEFILE_OPTION=--onefile
set DEBUG_OPTION=--debug=imports
set PATHS_OPTION=--paths=.

REM Additional PyInstaller options (modify as needed)
set EXTRA_OPTIONS=

REM Check if icon file exists
if exist "%ICON_FILE%" (
    echo Using icon: %ICON_FILE%
    set ICON_OPTION=--icon=%ICON_FILE%
) else (
    echo No icon file found. Skipping icon option.
    set ICON_OPTION=
)

REM Clean previous build directories if they exist
if exist "%BUILD_DIR%" (
    echo Removing previous build directory...
    rmdir /s /q "%BUILD_DIR%"
)
if exist "%DIST_DIR%" (
    echo Removing previous dist directory...
    rmdir /s /q "%DIST_DIR%"
)
if exist "%SPEC_FILE%" (
    echo Removing previous spec file...
    del "%SPEC_FILE%"
)

REM Run PyInstaller
echo Starting PyInstaller build...
pyinstaller %CLEAN_OPTION% %ONEFILE_OPTION% %DEBUG_OPTION% %PATHS_OPTION% %ICON_OPTION% %EXTRA_OPTIONS% --name=%OUTPUT_NAME% %MAIN_SCRIPT%

REM Check if the build was successful
if exist "%DIST_DIR%\%OUTPUT_NAME%.exe" (
    echo Build successful! Executable created: %DIST_DIR%\%OUTPUT_NAME%.exe
) else (
    echo Build failed. Check the logs for errors.
)

REM Pause to view output
pause
