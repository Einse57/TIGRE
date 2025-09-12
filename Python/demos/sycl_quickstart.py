"""
TIGRE SYCL Quick Start Example
==============================

This is a minimal example showing how to use TIGRE with Intel GPU acceleration
via the SYCL backend. Perfect for getting started quickly.
"""

import numpy as np
import tigre
import tigre.algorithms as algs

def quick_start_example():
    """
    Simple 5-minute example to get started with TIGRE + SYCL
    """
    print("TIGRE SYCL Quick Start")
    print("======================")
    
    # Step 1: Configure SYCL backend for Intel GPU acceleration
    print("1. Configuring Intel GPU backend...")
    tigre.configure_backend('sycl')
    print(f"   Active backend: {tigre.get_current_backend()}")
    
    # Step 2: Create CT geometry
    print("2. Creating CT geometry...")
    geo = tigre.geometry(mode='cone', nVoxel=np.array([64, 64, 64]))
    print(f"   Volume: {geo.nVoxel} voxels")
    
    # Step 3: Create a simple phantom
    print("3. Creating test phantom...")
    phantom = np.zeros(geo.nVoxel, dtype=np.float32)
    
    # Add a sphere in the center
    center = geo.nVoxel // 2
    y, x, z = np.ogrid[:geo.nVoxel[0], :geo.nVoxel[1], :geo.nVoxel[2]]
    sphere = ((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2) < (20)**2
    phantom[sphere] = 1.0
    
    # Step 4: Define scan angles
    print("4. Setting up scan...")
    angles = np.linspace(0, 2*np.pi, 180, dtype=np.float32)
    print(f"   Projections: {len(angles)}")
    
    # Step 5: Simulate projections (this uses Intel GPU!)
    print("5. Computing projections on Intel GPU...")
    projections = tigre.Ax(phantom, geo, angles)
    
    # Step 6: Reconstruct using FDK (this also uses Intel GPU!)
    print("6. Reconstructing with FDK on Intel GPU...")
    reconstructed = algs.fdk(projections, geo, angles)
    
    # Step 7: Check results
    print("7. Results:")
    print(f"   Original range: [{np.min(phantom):.3f}, {np.max(phantom):.3f}]")
    print(f"   Reconstructed range: [{np.min(reconstructed):.3f}, {np.max(reconstructed):.3f}]")
    print(f"   Mean squared error: {np.mean((reconstructed - phantom)**2):.6f}")
    
    print("\n✅ Quick start completed successfully!")
    print("   Your Intel GPU is now accelerating TIGRE reconstructions!")
    
    return phantom, projections, reconstructed

if __name__ == "__main__":
    # Run the quick start example
    phantom, projections, reconstructed = quick_start_example()
    
    print("\nNext steps:")
    print("- Try 'python sycl_examples.py' for more advanced examples")
    print("- See Frontispiece/SYCL_Backend.md for complete documentation")
    print("- Experiment with different algorithms: algs.cgls(), algs.sart(), etc.")