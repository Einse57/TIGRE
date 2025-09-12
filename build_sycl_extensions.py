"""
TIGRE SYCL Extension Builder
============================

This script builds the SYCL backend extensions for TIGRE Python bindings.
It automatically detects Intel oneAPI and configures the build environment.

Usage:
    python build_sycl_extensions.py

Requirements:
    - Intel oneAPI DPC++ Compiler 2025.1.1 or later
    - Intel oneAPI Base Toolkit
    - TIGRE with SYCL backend compiled (tigre_sycl.lib)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def detect_intel_oneapi():
    """Detect Intel oneAPI installation."""
    possible_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI",
        r"C:\Intel\oneAPI",
        "/opt/intel/oneapi"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return Path(path)
    
    # Try environment variable
    if "ONEAPI_ROOT" in os.environ:
        return Path(os.environ["ONEAPI_ROOT"])
    
    return None


def setup_intel_environment():
    """Setup Intel oneAPI environment."""
    oneapi_root = detect_intel_oneapi()
    
    if not oneapi_root:
        raise RuntimeError(
            "Intel oneAPI not found. Please install Intel oneAPI Base Toolkit.\n"
            "Download from: https://software.intel.com/content/www/us/en/develop/tools/oneapi.html"
        )
    
    print(f"Found Intel oneAPI at: {oneapi_root}")
    
    # Set up environment
    if os.name == 'nt':  # Windows
        setvars_script = oneapi_root / "setvars.bat"
        if setvars_script.exists():
            print("Initializing Intel oneAPI environment...")
            # On Windows, we need to run setvars.bat before building
            return str(setvars_script)
    else:  # Linux
        setvars_script = oneapi_root / "setvars.sh"
        if setvars_script.exists():
            print("Initializing Intel oneAPI environment...")
            return str(setvars_script)
    
    raise RuntimeError(f"Could not find setvars script in {oneapi_root}")


def check_sycl_library():
    """Check if TIGRE SYCL library exists."""
    sycl_lib = Path("tigre_sycl.lib")  # Windows
    if not sycl_lib.exists():
        sycl_lib = Path("libtigre_sycl.a")  # Linux
    
    if not sycl_lib.exists():
        raise RuntimeError(
            "TIGRE SYCL library not found (tigre_sycl.lib or libtigre_sycl.a).\n"
            "Please build the SYCL backend first:\n"
            "  mkdir build_sycl && cd build_sycl\n"
            "  cmake .. -DTIGRE_BUILD_SYCL=ON -DCMAKE_CXX_COMPILER=icx\n"
            "  cmake --build . --config Release"
        )
    
    print(f"Found TIGRE SYCL library: {sycl_lib}")
    return sycl_lib


def build_extensions():
    """Build the SYCL backend extensions."""
    print("Building SYCL backend extensions...")
    
    # Check requirements
    setvars_script = setup_intel_environment()
    sycl_lib = check_sycl_library()
    
    # Navigate to Python directory
    python_dir = Path("Python")
    if not python_dir.exists():
        raise RuntimeError("Python directory not found. Run from TIGRE root directory.")
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(python_dir)
        
        # Build command
        if os.name == 'nt':  # Windows
            # Create a batch script to source oneAPI and build
            build_script = """
@echo off
call "{setvars_script}"
python setup.py build_ext --inplace --with-sycl
""".format(setvars_script=setvars_script)
            
            with open("build_sycl_temp.bat", "w") as f:
                f.write(build_script)
            
            # Run the build
            result = subprocess.run(["build_sycl_temp.bat"], shell=True, capture_output=True, text=True)
            
            # Clean up
            try:
                os.remove("build_sycl_temp.bat")
            except:
                pass
                
        else:  # Linux
            # Source oneAPI environment and build
            build_command = f"source {setvars_script} && python setup.py build_ext --inplace --with-sycl"
            result = subprocess.run(build_command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        else:
            print("Build successful!")
            
            # Check if extensions were created
            extensions_dir = Path("tigre/utilities/cuda_interface")
            extensions = list(extensions_dir.glob("*backend*.pyd")) + list(extensions_dir.glob("*backend*.so"))
            
            if extensions:
                print("Created extensions:")
                for ext in extensions:
                    size = ext.stat().st_size / 1024  # KB
                    print(f"  - {ext.name} ({size:.1f} KB)")
                return True
            else:
                print("Warning: No extension files found after build.")
                return False
                
    finally:
        os.chdir(original_dir)


def test_extensions():
    """Test the built extensions."""
    print("Testing SYCL extensions...")
    
    try:
        # Simple import test
        import tigre
        
        # Try to configure SYCL backend
        tigre.configure_backend('sycl')
        backend = tigre.get_current_backend()
        
        if backend == 'sycl':
            print("✅ SYCL backend is working!")
            
            # Quick functionality test
            import numpy as np
            import tigre.algorithms as algs
            
            geo = tigre.geometry(mode='cone', nVoxel=np.array([32, 32, 32]))
            phantom = np.ones(geo.nVoxel, dtype=np.float32) * 0.5
            angles = np.linspace(0, 2*np.pi, 60, dtype=np.float32)
            
            projections = tigre.Ax(phantom, geo, angles)
            reconstructed = algs.fdk(projections, geo, angles)
            
            print(f"✅ Basic reconstruction test passed!")
            print(f"   Phantom shape: {phantom.shape}")
            print(f"   Reconstruction range: [{np.min(reconstructed):.3f}, {np.max(reconstructed):.3f}]")
            
            return True
            
        else:
            print(f"❌ SYCL backend not active. Current backend: {backend}")
            return False
            
    except Exception as e:
        print(f"❌ Extension test failed: {e}")
        return False


def main():
    """Main build process."""
    print("TIGRE SYCL Extension Builder")
    print("=" * 40)
    
    try:
        # Build extensions
        if build_extensions():
            print("\n" + "=" * 40)
            print("Build completed successfully!")
            
            # Test extensions
            print("\n" + "=" * 40)
            if test_extensions():
                print("\n🎉 TIGRE with SYCL backend is ready to use!")
                print("\nNext steps:")
                print("  - Try: python demos/sycl_quickstart.py")
                print("  - See: python demos/sycl_examples.py")
                print("  - Docs: Frontispiece/SYCL_Backend.md")
            else:
                print("\n⚠️ Build completed but testing failed.")
                print("Check your Intel oneAPI installation and GPU drivers.")
        else:
            print("\n❌ Build failed. Check error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()