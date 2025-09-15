#!/usr/bin/env python3
"""
TIGRE SYCL Installation Test
===========================

Quick verification script for TIGRE SYCL backend installation.
This performs basic sanity checks without running heavy computations.

Usage: python test_sycl_installation.py
"""

import sys
import traceback

def test_imports():
    """Test basic imports"""
    print("🔍 Testing Python imports...")
    
    try:
        import numpy as np
        print("  ✅ NumPy:", np.__version__)
    except ImportError:
        print("  ❌ NumPy not available")
        return False
    
    try:
        import tigre
        print("  ✅ TIGRE:", tigre.__version__ if hasattr(tigre, '__version__') else "imported")
    except ImportError as e:
        print(f"  ❌ TIGRE import failed: {e}")
        return False
    
    try:
        import tigre.algorithms as algs
        print("  ✅ TIGRE algorithms module")
    except ImportError as e:
        print(f"  ❌ TIGRE algorithms import failed: {e}")
        return False
    
    return True

def test_backend_loading():
    """Test SYCL backend loading"""
    print("\n🚀 Testing SYCL backend loading...")
    
    try:
        import tigre.utilities.Ax as Ax_backend
        print("  ✅ Ax backend module loaded")
    except ImportError as e:
        print(f"  ❌ Ax backend import failed: {e}")
        return False
    
    try:
        # Test if SYCL extension is available
        from tigre.utilities.cuda_interface import _Ax_backend
        print("  ✅ _Ax_backend extension loaded")
    except ImportError as e:
        print(f"  ❌ _Ax_backend extension failed: {e}")
        print("      This indicates SYCL extension build issues")
        return False
    
    return True

def test_geometry_creation():
    """Test basic geometry creation"""
    print("\n📐 Testing geometry creation...")
    
    try:
        from tigre.utilities.geometry import Geometry
        
        geo = Geometry()
        geo.nVoxel = [64, 64, 64]
        geo.sVoxel = [128, 128, 128]
        geo.nDetector = [128, 128]
        geo.sDetector = [256, 256]
        geo.DSD = 1536
        geo.DSO = 1000
        
        print("  ✅ Geometry object created successfully")
        print(f"      Voxels: {geo.nVoxel}")
        print(f"      Detector: {geo.nDetector}")
        return True
        
    except Exception as e:
        print(f"  ❌ Geometry creation failed: {e}")
        return False

def test_small_phantom():
    """Test small phantom creation and projection"""
    print("\n👻 Testing small phantom...")
    
    try:
        import numpy as np
        from tigre.utilities.geometry import Geometry
        
        # Very small test case
        geo = Geometry()
        geo.nVoxel = np.array([32, 32, 32])
        geo.sVoxel = np.array([64, 64, 64])
        geo.nDetector = np.array([64, 64])
        geo.sDetector = np.array([128, 128])
        geo.DSD = 1536
        geo.DSO = 1000
        
        # Simple phantom - sphere in center
        phantom = np.zeros(geo.nVoxel, dtype=np.float32)
        center = np.array(geo.nVoxel) // 2
        y, x, z = np.ogrid[:geo.nVoxel[0], :geo.nVoxel[1], :geo.nVoxel[2]]
        
        distance = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
        phantom[distance <= 8] = 1.0
        
        print(f"  ✅ Test phantom created: {phantom.shape}")
        print(f"      Non-zero voxels: {np.count_nonzero(phantom)}")
        
        return True, geo, phantom
        
    except Exception as e:
        print(f"  ❌ Phantom creation failed: {e}")
        return False, None, None

def test_sycl_projection():
    """Test SYCL forward projection"""
    print("\n⚡ Testing SYCL forward projection...")
    
    success, geo, phantom = test_small_phantom()
    if not success:
        return False
    
    try:
        import numpy as np
        import tigre
        
        # Single angle test
        angles = np.array([0.0])
        
        print("  🔄 Attempting SYCL projection...")
        projections = tigre.Ax(phantom, geo, angles, backend='sycl')
        
        print(f"  ✅ SYCL projection completed!")
        print(f"      Projection shape: {projections.shape}")
        print(f"      Projection range: [{np.min(projections):.3f}, {np.max(projections):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SYCL projection failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        
        # Check for common issues
        if "timeout" in str(e).lower():
            print("      → This may be a GPU timeout issue")
            print("        Try setting SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0")
        elif "device" in str(e).lower():
            print("      → This may be a device selection issue")
            print("        Check 'sycl-ls' output for available devices")
        elif "memory" in str(e).lower():
            print("      → This may be a memory issue")
            print("        Try reducing phantom/detector size")
        
        return False

def test_device_info():
    """Test device information gathering"""
    print("\n💻 Checking device information...")
    
    import subprocess
    import os
    
    # Check if sycl-ls is available
    try:
        result = subprocess.run(['sycl-ls'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ SYCL device listing:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"      {line}")
        else:
            print("  ❌ sycl-ls command failed")
            print(f"      stderr: {result.stderr}")
            
    except FileNotFoundError:
        print("  ❌ sycl-ls command not found")
        print("      Make sure Intel oneAPI environment is active")
    except subprocess.TimeoutExpired:
        print("  ❌ sycl-ls command timed out")
    except Exception as e:
        print(f"  ❌ Device listing failed: {e}")
    
    # Check environment variables
    sycl_vars = ['SYCL_DEVICE_FILTER', 'SYCL_RT_WARNING_LEVEL']
    env_found = False
    for var in sycl_vars:
        if var in os.environ:
            print(f"  🔧 {var}={os.environ[var]}")
            env_found = True
    
    if not env_found:
        print("  ℹ️  No SYCL environment variables set (this is normal)")

def main():
    """Run all tests"""
    print("🎯 TIGRE SYCL Installation Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Backend Loading", test_backend_loading), 
        ("Geometry Creation", test_geometry_creation),
        ("SYCL Projection", test_sycl_projection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Device info (informational only)
    test_device_info()
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("-" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏆 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! TIGRE SYCL backend appears to be working correctly.")
        print("\nNext steps:")
        print("• Run full demo: python examples/tigre_sycl_demo.py")
        print("• Try your own reconstructions with backend='sycl'")
        print("• See README_SYCL.md for usage guide")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("• Ensure Intel oneAPI 2025.2+ is installed and active")
        print("• Update Intel GPU drivers to latest version")
        print("• Rebuild SYCL backend: python build_tigre_sycl.py")
        print("• Check README_SYCL.md troubleshooting section")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())