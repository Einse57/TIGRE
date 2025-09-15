# TIGRE SYCL Backend: Intel GPU Acceleration Guide

## Overview

TIGRE's SYCL backend provides high-performance computed tomography reconstruction using Intel GPU acceleration through the Intel oneAPI SYCL programming model. This enables significant speedups for CT reconstruction algorithms on Intel integrated and discrete graphics hardware.

## 🚀 Quick Start

### Prerequisites

- **Intel oneAPI Base Toolkit 2025.2+** (required for SYCL compiler and runtime)
- **Intel GPU drivers 32.0.101.7000+** (for GPU device support)
- **Python 3.12+** with NumPy and Cython
- **Visual Studio 2022** with C++ support (Windows)

### Installation

1. **Download and install Intel oneAPI Base Toolkit**:
   ```
   https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html
   ```

2. **Update Intel GPU drivers**:
   ```
   https://www.intel.com/content/www/us/en/support/products/80939/graphics.html
   ```

3. **Build TIGRE with SYCL support**:
   ```bash
   python build_tigre_sycl.py
   ```

4. **Test the installation**:
   ```bash
   tigre_sycl_env.bat
   python examples/tigre_sycl_demo.py
   ```

## 🔧 Building from Source

### Automated Build (Recommended)

The provided build script handles all compilation steps:

```bash
# Full build
python build_tigre_sycl.py

# Clean previous builds
python build_tigre_sycl.py clean
```

### Manual Build Steps

If you need to customize the build process:

1. **Set up Intel oneAPI environment**:
   ```cmd
   call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   ```

2. **Compile SYCL backend library**:
   ```cmd
   icx -c -fsycl /std:c++17 /EHsc ^
       -I"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" ^
       -I"Common\CUDA" ^
       src\backends\sycl\*.cpp
   ```

3. **Create static library**:
   ```cmd
   lib /OUT:tigre_sycl_backend.lib *.obj
   ```

4. **Build Python extensions**:
   ```bash
   python setup.py build_ext --inplace
   ```

## 📚 Usage Guide

### Basic Usage

```python
import tigre
import tigre.algorithms as algs
import numpy as np

# Create geometry and phantom
geo = tigre.geometry()
phantom = tigre.sample_loader.load_head_phantom()
angles = np.linspace(0, 2*np.pi, 180)

# Forward projection with SYCL backend
projections = tigre.Ax(phantom, geo, angles, backend='sycl')

# FDK reconstruction with SYCL backend
reconstruction = algs.fdk(projections, geo, angles, backend='sycl')
```

### Backend Selection

TIGRE automatically detects available backends. You can explicitly select:

```python
# Available backends: 'auto', 'cpu', 'sycl'
projections = tigre.Ax(phantom, geo, angles, backend='sycl')
```

### Supported Algorithms

The SYCL backend currently supports:

- **Forward Projection**: `tigre.Ax()` with Siddon ray-tracing
- **FDK Reconstruction**: `algs.fdk()` with cone-beam filtered backprojection
- **CGLS**: `algs.cgls()` conjugate gradient least squares
- **ASD-POCS**: `algs.asd_pocs()` adaptive steepest descent

More algorithms are being ported to SYCL backend.

### Performance Tuning

1. **GPU Memory Management**:
   ```python
   # For large datasets, process in chunks
   chunk_size = 50  # projections per chunk
   for i in range(0, len(angles), chunk_size):
       chunk_angles = angles[i:i+chunk_size]
       chunk_proj = tigre.Ax(phantom, geo, chunk_angles, backend='sycl')
   ```

2. **Optimize Geometry Settings**:
   ```python
   # Balance quality vs speed
   geo.nVoxel = [256, 256, 256]     # Start with moderate resolution
   geo.nDetector = [512, 512]       # Adjust based on GPU memory
   ```

## 🔍 Device Information

### Check Available Devices

```bash
# Intel SYCL device listing
sycl-ls

# Expected output:
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM)
[opencl:cpu:1] Intel(R) OpenCL
[opencl:gpu:2] Intel(R) Graphics [0x4682]
[level_zero:gpu:0] Intel(R) Graphics [0x4682]
```

### Runtime Device Selection

SYCL runtime automatically selects the best available device. For manual control:

```python
import os
# Force specific device (0=CPU, 1=GPU)
os.environ['SYCL_DEVICE_FILTER'] = 'level_zero:gpu'
```

## ⚡ Performance Expectations

### Typical Speedups (vs CPU)

| Operation | Dataset Size | Intel GPU | Speedup |
|-----------|--------------|-----------|---------|
| Forward Projection | 256³ phantom, 180 projections | Iris Xe | 3-5x |
| FDK Reconstruction | 512² detector, 180 projections | Iris Xe | 4-7x |
| CGLS (10 iterations) | 256³ volume | Iris Xe | 2-4x |

*Performance varies by hardware, dataset size, and system configuration.*

### Memory Requirements

- **GPU Memory**: ~4GB recommended for 256³ reconstructions
- **System Memory**: 16GB+ for large datasets
- **Shared GPU Memory**: Intel GPUs use system RAM when needed

## 🛠️ Troubleshooting

### Common Issues

1. **"sycl-ls" shows no GPU devices**:
   - Update Intel GPU drivers
   - Install Intel Graphics Driver from Intel website
   - Reboot after driver installation

2. **Import errors for _Ax_backend**:
   ```bash
   # Rebuild Python extensions
   python build_tigre_sycl.py clean
   python build_tigre_sycl.py
   ```

3. **GPU execution hangs or times out**:
   ```python
   # Set longer timeout
   import os
   os.environ['SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS'] = '0'
   ```

4. **"Intel oneAPI not found" error**:
   - Install Intel oneAPI Base Toolkit 2025.2+
   - Ensure installation path is standard location
   - Run `tigre_sycl_env.bat` to set environment

### Debug Mode

Enable detailed SYCL debugging:

```bash
set SYCL_RT_WARNING_LEVEL=2
set SYCL_PI_TRACE=-1
python your_script.py
```

### Performance Debugging

```python
# Enable profiling
import os
os.environ['SYCL_ENABLE_PER_KERNEL_PROFILING'] = '1'

# Time individual operations
import time
start = time.time()
result = tigre.Ax(phantom, geo, angles, backend='sycl')
print(f"Forward projection: {time.time() - start:.2f}s")
```

## 🔬 Advanced Usage

### Custom Geometries

```python
# High-resolution cardiac CT
geo = tigre.geometry()
geo.nVoxel = np.array([512, 512, 300])
geo.sVoxel = np.array([400, 400, 300])  # mm
geo.nDetector = np.array([1024, 768])
geo.sDetector = np.array([400, 300])
geo.DSD = 1200  # Source-detector distance
geo.DSO = 800   # Source-object distance
```

### Iterative Reconstruction

```python
# CGLS with SYCL acceleration
reconstruction = algs.cgls(projections, geo, angles, 
                          niter=20, 
                          backend='sycl',
                          verbose=True)

# ASD-POCS with TV regularization  
reconstruction = algs.asd_pocs(projections, geo, angles,
                              niter=50,
                              alpha=0.002,
                              backend='sycl')
```

### Batch Processing

```python
def process_scan_batch(scan_files, output_dir):
    """Process multiple CT scans with SYCL acceleration"""
    for scan_file in scan_files:
        projections = load_projections(scan_file)
        
        # Reconstruct with SYCL
        recon = algs.fdk(projections, geo, angles, backend='sycl')
        
        # Save result
        output_file = output_dir / f"recon_{scan_file.stem}.npy"
        np.save(output_file, recon)
```

## 📊 Benchmarking

### Performance Comparison Script

```python
import time
import numpy as np

def benchmark_backends(phantom, geo, angles):
    """Compare CPU vs SYCL performance"""
    results = {}
    
    for backend in ['cpu', 'sycl']:
        start = time.time()
        proj = tigre.Ax(phantom, geo, angles, backend=backend)
        recon = algs.fdk(proj, geo, angles, backend=backend)
        total_time = time.time() - start
        
        results[backend] = {
            'time': total_time,
            'projection_shape': proj.shape,
            'reconstruction_shape': recon.shape
        }
    
    speedup = results['cpu']['time'] / results['sycl']['time']
    print(f"SYCL Speedup: {speedup:.1f}x")
    
    return results
```

## 🤝 Contributing

### Development Setup

1. Fork the TIGRE repository
2. Create feature branch: `git checkout -b sycl-feature`
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Build and test your changes
5. Submit pull request

### Adding SYCL Algorithm Support

To port a new algorithm to SYCL:

1. Create SYCL implementation in `src/backends/sycl/`
2. Add factory registration in `device_backend_factory.cpp`
3. Update Python bindings in `_Ax_backend.pyx`
4. Add tests and documentation

## 📝 License

TIGRE SYCL backend follows the same BSD license as TIGRE.

## 🙏 Acknowledgments

- Intel oneAPI team for SYCL compiler and runtime
- TIGRE development community
- Open source contributors

## 📞 Support

- **Documentation**: https://github.com/CERN/TIGRE/wiki
- **Issues**: https://github.com/CERN/TIGRE/issues  
- **Discussions**: https://github.com/CERN/TIGRE/discussions
- **SYCL Resources**: https://intel.github.io/llvm-docs/

---

*Last updated: December 2024 | TIGRE SYCL Backend v1.0*