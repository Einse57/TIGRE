@echo off
REM TIGRE SYCL Environment Launcher
REM Automatically sets up Intel oneAPI environment for TIGRE with SYCL backend

echo.
echo ========================================
echo   TIGRE SYCL Backend Environment
echo ========================================
echo   Intel GPU acceleration ready
echo   Intel oneAPI 2025.2+ required
echo.

echo Initializing Intel oneAPI environment...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1

if %errorlevel% neq 0 (
    echo ❌ Intel oneAPI not found or failed to initialize
    echo    Please install Intel oneAPI Base Toolkit 2025.2+
    echo    Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html
    pause
    exit /b 1
)

echo ✅ Intel oneAPI environment initialized

echo.
echo TIGRE SYCL Backend Ready!
echo Available backends: CPU, SYCL (Intel GPU)
echo.
echo Quick Start Commands:
echo   python test_sycl_installation.py     # Test installation
echo   python examples/tigre_sycl_demo.py   # Run full demo
echo   python -c "import tigre; print('TIGRE version:', getattr(tigre, '__version__', 'dev'))"
echo.
echo Example Usage:
echo   import tigre
echo   projections = tigre.Ax(volume, geo, angles, backend='sycl')
echo   reconstruction = tigre.algorithms.fdk(projections, geo, angles, backend='sycl')
echo.

cmd /k "cd /d %~dp0 && echo Ready for TIGRE with Intel GPU acceleration!"