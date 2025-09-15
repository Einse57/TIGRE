"""
SYCL Runtime Configuration for TIGRE

This module ensures Intel oneAPI runtime libraries are properly loaded
for SYCL backend extensions on Windows.
"""

import os
import sys
import platform


def configure_sycl_runtime():
    """
    Configure the runtime environment for SYCL extensions.
    
    On Windows, this adds Intel oneAPI DLL directories to the DLL search path
    to ensure SYCL extensions can load their dependencies.
    """
    if platform.system() != 'Windows':
        # Only needed on Windows for DLL loading
        return []
    
    # Intel oneAPI DLL paths that SYCL extensions need
    intel_dll_paths = [
        'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin',
        'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib', 
        'C:/Program Files (x86)/Intel/oneAPI/tbb/latest/bin',
        'C:/Program Files (x86)/Intel/oneAPI/tbb/latest/lib',
        'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/bin',
        'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib'
    ]
    
    # Add existing paths to DLL search
    added_paths = []
    for path in intel_dll_paths:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
                added_paths.append(path)
            except (OSError, AttributeError):
                # add_dll_directory not available or failed
                # Fallback to PATH modification
                current_path = os.environ.get('PATH', '')
                if path not in current_path:
                    os.environ['PATH'] = path + os.pathsep + current_path
                added_paths.append(path + " (via PATH)")
    
    return added_paths


def ensure_sycl_backend_available():
    """
    Ensure SYCL backend extensions can be imported.
    
    Returns:
        bool: True if SYCL backend is available, False otherwise
    """
    # Configure runtime first
    configure_sycl_runtime()
    
    try:
        # Test import of SYCL extensions
        import _Ax_backend
        import _Atb_backend
        return True
    except ImportError:
        return False


# Auto-configure when module is imported
if __name__ != '__main__':
    configure_sycl_runtime()