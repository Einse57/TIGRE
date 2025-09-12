# TIGRE SYCL Backend Integration

## Overview

TIGRE now supports GPU acceleration through the SYCL (SYCL C++) backend, enabling high-performance computed tomography reconstruction non-CUDA GPUs, integrated graphics, and other SYCL-compatible devices. This integration provides a multi-backend architecture that supports CUDA, SYCL, and CPU execution modes.

## Features

- **Broad GPU Acceleration**: Native support for SYCL GPUs and integrated graphics
- **Multi-Backend Architecture**: Switches between CUDA, SYCL, and CPU backends
- **TIGRE Compatibility**: Support for all TIGRE reconstruction algorithms
- **Python Integration**: Full Python binding support with automatic backend selection
- **Memory Optimization**: GPU memory management for large datasets

## Requirements

### Hardware
- GPU or integrated graphics with SYCL support
- Minimum 4GB GPU memory for typical reconstruction tasks
- CPU with integrated graphics (backup option)

### Software
- **Intel oneAPI DPC++ Compiler 2025.1.1 or later**
- **Intel oneAPI Base Toolkit** (includes SYCL runtime)
- **Python 3.8-3.12** (for Python bindings)
- **CMake 3.20 or later**
- **Visual Studio 2019/2022** (Windows) or appropriate C++ compiler

## Installation

### Step 1: Install Intel oneAPI

1. Download Intel oneAPI Base Toolkit from [Intel Developer Zone](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html)
2. Install the toolkit (includes DPC++ compiler and SYCL runtime)
3. Initialize the environment:
   ```bash
   # Windows
   "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   
   # Linux
   source /opt/intel/oneapi/setvars.sh
   ```

### Step 2: Build TIGRE with SYCL Backend

```bash
# Clone TIGRE repository
git clone https://github.com/CERN/TIGRE.git
cd TIGRE

# Create build directory
mkdir build_sycl
cd build_sycl

# Configure with SYCL backend
cmake .. -DTIGRE_BUILD_SYCL=ON -DCMAKE_CXX_COMPILER=icx

# Build the project
cmake --build . --config Release

# Install (optional)
cmake --install . --config Release
```

### Step 3: Build Python Bindings

```bash
cd ../Python

# Create and activate virtual environment
python -m venv tigre_sycl_env
# Windows
tigre_sycl_env\Scripts\activate
# Linux
source tigre_sycl_env/bin/activate

# Install dependencies
pip install cython numpy scipy matplotlib

# Build extensions with SYCL support
python setup.py build_ext --inplace --with-sycl

# Install TIGRE
pip install -e .
```

## Usage

### Python API

The SYCL backend integrates seamlessly with TIGRE's existing Python API:

```python
import tigre
import tigre.algorithms as algs
import numpy as np

# Create geometry (same as always)
geo = tigre.geometry(mode='cone', nVoxel=np.array([128, 128, 128]))

# Configure SYCL backend
tigre.configure_backend('sycl')  # Options: 'cuda', 'sycl', 'cpu'

# Generate sample data
angles = np.linspace(0, 2*np.pi, 360)
projections = tigre.Ax(np.ones(geo.nVoxel), geo, angles)

# Reconstruct using any TIGRE algorithm
reconstructed = algs.fdk(projections, geo, angles)

# The algorithm automatically uses Intel GPU acceleration!
```

### Backend Selection

TIGRE automatically selects the best available backend, but you can specify:

```python
# Explicit backend selection
tigre.configure_backend('sycl')    # Force SYCL/Intel GPU
tigre.configure_backend('cuda')    # Force CUDA/NVIDIA GPU  
tigre.configure_backend('cpu')     # Force CPU execution

# Query available backends
backends = tigre.list_backends()
print(f"Available backends: {backends}")

# Check current backend
current = tigre.get_current_backend()
print(f"Using backend: {current}")
```

### Performance Optimization

```python
# Optimize for Intel GPU memory
tigre.set_gpu_memory_policy('conservative')  # Better for limited GPU memory
tigre.set_gpu_memory_policy('aggressive')    # Better for high-memory GPUs

# Batch processing for large datasets
batch_size = tigre.get_optimal_batch_size(geo, device='sycl')
for i in range(0, len(angles), batch_size):
    batch_angles = angles[i:i+batch_size]
    batch_projections = projections[:, :, i:i+batch_size]
    # Process batch...
```

## Supported Algorithms

All TIGRE algorithms are supported with SYCL backend:

### Analytical Algorithms
- **FDK**: Feldkamp-Davis-Kress reconstruction
- **FBP**: Filtered Backprojection

### Iterative Algorithms
- **ART Family**: ART, SART, OS-SART
- **CGLS**: Conjugate Gradient Least Squares
- **SIRT**: Simultaneous Iterative Reconstruction Technique

### Advanced Algorithms
- **TV Regularization**: Total Variation minimization
- **PICCS**: Prior Image Constrained Compressed Sensing
- **Statistical Algorithms**: MLEM, OSEM

## Troubleshooting

### Common Issues

**1. "SYCL runtime not found"**
```bash
# Ensure Intel oneAPI is properly installed and initialized
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"  # Windows
source /opt/intel/oneapi/setvars.sh                # Linux
```

**2. "No SYCL devices available"**
```python
# Check available SYCL devices
import tigre.utilities.sycl_utils as sycl
devices = sycl.list_sycl_devices()
print("Available SYCL devices:", devices)
```

**3. "Compilation errors with extensions"**
```bash
# Ensure correct compiler is used
export CXX=icx  # Linux
# or use CMake flag: -DCMAKE_CXX_COMPILER=icx
```

**4. "Out of memory errors"**
```python
# Reduce problem size or use memory optimization
tigre.set_gpu_memory_policy('conservative')
geo.nVoxel = np.array([64, 64, 64])  # Reduce volume size
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import tigre
tigre.set_log_level('debug')  # Options: 'debug', 'info', 'warning', 'error'
```

## Development and Contributing

### Building from Source

```bash
# Development build with debug symbols
cmake .. -DTIGRE_BUILD_SYCL=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=icx
cmake --build . --config Debug
```

### Running Tests

```bash
# C++ tests
cd build_sycl
ctest --output-on-failure

# Python tests
cd ../Python
python -m pytest tests/ -v --tb=short
```

### Code Structure

```
Common/SYCL/           # SYCL backend implementation
├── sycl_backend.cpp   # Main backend interface
├── sycl_projections.cpp # Forward projection kernels
├── sycl_backprojection.cpp # Backprojection kernels
└── sycl_utils.cpp     # Utilities and device management

Python/tigre/utilities/sycl_interface/  # Python bindings
├── _Ax_backend.pyx    # Forward projection wrapper
├── _Atb_backend.pyx   # Backprojection wrapper
└── sycl_config.py     # Configuration utilities
```

## License

TIGRE SYCL backend integration maintains the same license terms as the main TIGRE project. See [LICENSE.txt](../LICENSE.txt) for details.

## Citation

If you use TIGRE with SYCL backend in your research, please cite:

```bibtex
@article{biguri2016tigre,
  title={TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction},
  author={Biguri, Ander and Dosanjh, Manjit and Hancock, Steven and Soleimani, Manuchehr},
  journal={Biomedical Physics \& Engineering Express},
  volume={2},
  number={5},
  pages={055010},
  year={2016},
  publisher={IOP Publishing}
}
```

## Support

- **Issues**: Report bugs and issues on [GitHub Issues](https://github.com/CERN/TIGRE/issues)
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/CERN/TIGRE/discussions)
- **Intel oneAPI Support**: [Intel Developer Zone](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html)

---

*Last updated: September 2025*