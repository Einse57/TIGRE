#!/usr/bin/env python3#!/usr/bin/env python3

""""""

TIGRE SYCL Backend BuilderBuild TIGRE with SYCL backend support.

==========================

This script builds TIGRE with minimal changes to preserve existing functionality

This script builds TIGRE with Intel GPU SYCL backend support for high-performancewhile adding Intel GPU/SYCL support via the existing backend architecture.

computed tomography reconstruction algorithms."""



Requirements:import os

- Python 3.12+import sys

- Intel oneAPI 2025.2 or laterimport subprocess

- Visual Studio 2022 with C++ supportimport shutil

- Updated Intel GPU driversfrom pathlib import Path



Usage:

    python build_tigre_sycl.pydef initialize_oneapi():

    """Initialize Intel oneAPI environment."""

This will:    print("🔧 Initializing Intel oneAPI environment...")

1. Check Intel oneAPI installation    

2. Build TIGRE backend library with SYCL support    oneapi_root = r"C:\Program Files (x86)\Intel\oneAPI"

3. Compile Python extensions    if not os.path.exists(oneapi_root):

4. Set up runtime environment        raise RuntimeError("Intel oneAPI not found. Please install Intel oneAPI Base Toolkit.")

"""    

    # Add oneAPI to PATH

import os    oneapi_bin = os.path.join(oneapi_root, "compiler", "latest", "bin")

import sys    if oneapi_bin not in os.environ.get('PATH', ''):

import shutil        os.environ['PATH'] += f";{oneapi_bin}"

import subprocess        

import numpy as np    print(f"✅ oneAPI environment ready: {oneapi_bin}")

from pathlib import Path

from setuptools import setup, Extension

from Cython.Build import cythonizedef build_sycl_backend():

    """Build the SYCL backend components."""

class TIGRESYCLBuilder:    print("🏗️  Building SYCL backend...")

    def __init__(self):    

        self.tigre_root = Path(__file__).parent.absolute()    # Create SYCL library from source files

        self.python_dir = self.tigre_root / "Python"    sycl_sources = [

                "src/backends/sycl/device_backend_sycl.cpp",

        # Intel oneAPI paths        "src/backends/sycl/siddon_projection_sycl.cpp", 

        self.intel_root = Path(r"C:\Program Files (x86)\Intel\oneAPI")        "src/backends/sycl/fdk_backprojection_sycl.cpp"

        self.intel_compiler = self.intel_root / "compiler" / "latest"    ]

            

    def check_requirements(self):    # Check that source files exist

        """Check if all requirements are met"""    for src in sycl_sources:

        print("🔍 Checking requirements...")        if not os.path.exists(src):

                    print(f"⚠️  Source file not found: {src}")

        # Check Intel oneAPI            return False

        if not (self.intel_root / "setvars.bat").exists():    

            print("❌ Intel oneAPI not found!")    # Compile SYCL backend

            print("   Please install Intel oneAPI Base Toolkit 2025.2 or later")    include_dirs = [

            print("   Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html")        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include",

            return False        "include",

                "Common/CUDA"  # For TIGRE headers

        # Check compiler    ]

        icx_exe = self.intel_compiler / "bin" / "icx.exe"    

        if not icx_exe.exists():    compile_cmd = [

            print("❌ Intel DPC++ compiler not found!")        "icx",  # Intel DPC++ compiler

            return False        "-c",   # Compile only

                "-std=c++17",

        # Check Python packages        "-DTIGRE_BUILD_SYCL_BACKEND=1"

        try:    ]

            import numpy    

            import Cython    # Add include directories

        except ImportError as e:    for inc_dir in include_dirs:

            print(f"❌ Required Python package missing: {e}")        compile_cmd.extend(["-I", inc_dir])

            print("   Please install: pip install numpy cython")    

            return False    # Add source files

            compile_cmd.extend(sycl_sources)

        print("✅ All requirements met")    

        return True    print(f"Running: {' '.join(compile_cmd)}")

        

    def build_backend_library(self):    try:

        """Build TIGRE backend library with SYCL support"""        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)

        print("🔨 Building TIGRE backend library...")        print("✅ SYCL sources compiled successfully")

                

        src_dir = self.tigre_root / "src" / "backends"        # Create static library

                lib_cmd = ["lib", "/OUT:tigre_sycl.lib"] + [f.replace('.cpp', '.obj') for f in [os.path.basename(s) for s in sycl_sources]]

        # Source files to compile        subprocess.run(lib_cmd, check=True)

        source_files = [        print("✅ SYCL library created: tigre_sycl.lib")

            src_dir / "device_backend_factory.cpp",        

            src_dir / "sycl" / "device_backend_sycl.cpp",        # Clean up object files

            src_dir / "sycl" / "siddon_projection_sycl_fixed.cpp",        for src in sycl_sources:

            src_dir / "sycl" / "fdk_backprojection_sycl.cpp",            obj_file = os.path.basename(src).replace('.cpp', '.obj')

        ]            if os.path.exists(obj_file):

                        os.remove(obj_file)

        # Compile each source file                

        icx_exe = self.intel_compiler / "bin" / "icx.exe"        return True

        object_files = []        

            except subprocess.CalledProcessError as e:

        for src_file in source_files:        print(f"❌ Compilation failed: {e}")

            if not src_file.exists():        print(f"stdout: {e.stdout}")

                print(f"❌ Source file not found: {src_file}")        print(f"stderr: {e.stderr}")

                return False        return False

                

            obj_file = src_file.with_suffix('.obj')

            print(f"   Compiling: {src_file.name}")def build_tigre_python():

                """Build TIGRE Python extensions with SYCL support."""

            cmd = [    print("🐍 Building TIGRE Python extensions...")

                str(icx_exe), "-c", "-fsycl", "/std:c++17", "/EHsc",    

                f"-I{self.intel_compiler / 'include'}",    try:

                f"-I{self.tigre_root / 'include'}",        # Build with SYCL support

                f"-I{self.tigre_root / 'Common' / 'CUDA'}",        build_cmd = [

                str(src_file), f"/Fo{obj_file}"            sys.executable, "setup.py", 

            ]            "build_ext", "--inplace",

                        "--with-sycl"  # Enable SYCL backend

            result = subprocess.run(cmd, capture_output=True, text=True)        ]

            if result.returncode != 0:        

                print(f"❌ Compilation failed for {src_file.name}")        print(f"Running: {' '.join(build_cmd)}")

                print(result.stderr)        result = subprocess.run(build_cmd, check=True)

                return False        print("✅ TIGRE Python extensions built successfully")

                    return True

            object_files.append(obj_file)        

            except subprocess.CalledProcessError as e:

        # Create static library        print(f"❌ Python extension build failed: {e}")

        print("   Creating backend library...")        return False

        lib_file = self.tigre_root / "tigre_sycl_backend.lib"

        

        lib_cmd = [def install_tigre():

            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx86\x64\lib.exe",    """Install TIGRE with SYCL support."""

            "/OUT:" + str(lib_file)    print("📦 Installing TIGRE...")

        ] + [str(obj) for obj in object_files]    

            try:

        result = subprocess.run(lib_cmd, capture_output=True, text=True)        install_cmd = [sys.executable, "setup.py", "install", "--with-sycl"]

        if result.returncode != 0:        subprocess.run(install_cmd, check=True)

            print("❌ Library creation failed")        print("✅ TIGRE installed successfully")

            print(result.stderr)        return True

            return False    except subprocess.CalledProcessError as e:

                print(f"❌ TIGRE installation failed: {e}")

        print(f"✅ Backend library created: {lib_file}")        return False

        return True

    

    def build_python_extensions(self):def main():

        """Build Python Cython extensions"""    """Build TIGRE with SYCL backend support."""

        print("🐍 Building Python extensions...")    

            print("🚀 TIGRE SYCL Backend Builder")

        # Extension settings    print("=" * 40)

        extensions = [    print("Preserving TIGRE functionality while adding Intel GPU support")

            Extension(    print()

                "_Ax_backend",    

                [str(self.python_dir / "tigre" / "utilities" / "cuda_interface" / "_Ax_backend.pyx")],    try:

                include_dirs=[        # Step 1: Initialize oneAPI

                    str(self.intel_compiler / "include" / "sycl"),        initialize_oneapi()

                    str(self.intel_compiler / "include"),        

                    str(self.tigre_root / "Common" / "CUDA"),        # Step 2: Build SYCL backend

                    str(self.tigre_root / "include"),        if not build_sycl_backend():

                    numpy.get_include(),            print("❌ Failed to build SYCL backend")

                ],            return 1

                library_dirs=[str(self.intel_compiler / "lib")],            

                libraries=["sycl8", "libircmd"],        # Step 3: Build Python extensions 

                extra_objects=[str(self.tigre_root / "tigre_sycl_backend.lib")],        if not build_tigre_python():

                language="c++",            print("❌ Failed to build Python extensions")

                extra_compile_args=["/std:c++17"],            return 1

            ),            

        ]        # Step 4: Install TIGRE

                if not install_tigre():

        # Build extensions            print("❌ Failed to install TIGRE")

        try:            return 1

            setup(            

                ext_modules=cythonize(extensions, language_level=3),        print("\n🎉 TIGRE with SYCL backend built successfully!")

                zip_safe=False,        print("You can now use:")

                script_args=['build_ext', '--inplace']        print("  import tigre")

            )        print("  from tigre.utilities.Ax_backend import Ax")

            print("✅ Python extensions built successfully")        print("  projections = Ax(phantom, geo, angles, backend='SYCL')")

            return True        

        except Exception as e:        return 0

            print(f"❌ Extension build failed: {e}")        

            return False    except Exception as e:

            print(f"❌ Build failed: {e}")

    def setup_runtime(self):        return 1

        """Set up runtime environment"""

        print("⚙️ Setting up runtime environment...")

        if __name__ == "__main__":

        # Copy required Intel runtime DLLs to Python directory    sys.exit(main())
        intel_bin = self.intel_compiler / "bin"
        required_dlls = ["sycl8.dll", "libircmd.dll", "ur_win_proxy_loader.dll"]
        
        copied = 0
        for dll in required_dlls:
            src = intel_bin / dll
            dst = self.python_dir / dll
            
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception as e:
                    print(f"   Warning: Failed to copy {dll}: {e}")
        
        print(f"✅ Copied {copied}/{len(required_dlls)} runtime DLLs")
        return True
    
    def create_launcher(self):
        """Create convenient launcher script"""
        print("🚀 Creating launcher script...")
        
        launcher_content = f'''@echo off
REM TIGRE SYCL Environment Launcher
REM Automatically sets up Intel oneAPI environment for TIGRE with SYCL backend

echo Initializing Intel oneAPI environment...
call "{self.intel_root}\\setvars.bat" > nul 2>&1

echo TIGRE SYCL Backend Ready!
echo Available backends: CPU, SYCL (Intel GPU)
echo.
echo Example usage:
echo   python -c "import tigre; print('TIGRE with SYCL:', tigre.__version__)"
echo.

cmd /k
'''
        
        launcher_file = self.tigre_root / "tigre_sycl_env.bat"
        with open(launcher_file, 'w') as f:
            f.write(launcher_content)
        
        print(f"✅ Launcher created: {launcher_file}")
        return True
    
    def build_all(self):
        """Build complete TIGRE SYCL backend"""
        print("🎯 Building TIGRE with Intel GPU SYCL Backend")
        print("=" * 50)
        
        steps = [
            ("Check Requirements", self.check_requirements),
            ("Build Backend Library", self.build_backend_library),
            ("Build Python Extensions", self.build_python_extensions),
            ("Setup Runtime", self.setup_runtime),
            ("Create Launcher", self.create_launcher),
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"\n❌ Build failed at: {step_name}")
                return False
        
        print(f"\n🎉 TIGRE SYCL backend build completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Run: tigre_sycl_env.bat")
        print(f"2. Test: python examples/tigre_sycl_demo.py")
        print(f"3. See: README_SYCL.md for usage guide")
        
        return True

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        print("🧹 Cleaning build artifacts...")
        # Clean up build artifacts
        for pattern in ["*.obj", "*.lib", "*.pyd", "build/"]:
            for f in Path(".").rglob(pattern):
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
        print("✅ Clean complete")
        return
    
    builder = TIGRESYCLBuilder()
    success = builder.build_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()