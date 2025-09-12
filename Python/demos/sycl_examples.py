"""
TIGRE SYCL Backend Examples
===========================

This module demonstrates practical usage of TIGRE with Intel GPU acceleration
via the SYCL backend. These examples show how to leverage Intel Arc GPUs and
integrated graphics for high-performance computed tomography reconstruction.

Examples included:
1. Basic FDK reconstruction with SYCL
2. Iterative reconstruction (CGLS) with memory optimization
3. Multi-backend comparison and benchmarking
4. Large dataset processing with batch optimization
5. Custom reconstruction pipeline with SYCL
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tigre
import tigre.algorithms as algs


def example_1_basic_fdk_sycl():
    """
    Example 1: Basic FDK reconstruction using SYCL backend
    
    This example demonstrates the simplest way to use TIGRE with Intel GPU
    acceleration for standard FDK reconstruction.
    """
    print("=" * 60)
    print("Example 1: Basic FDK Reconstruction with SYCL")
    print("=" * 60)
    
    # Configure SYCL backend
    print("Configuring SYCL backend...")
    tigre.configure_backend('sycl')
    print(f"Active backend: {tigre.get_current_backend()}")
    
    # Create geometry for a cone beam CT
    geo = tigre.geometry(mode='cone', nVoxel=np.array([128, 128, 128]))
    geo.dVoxel = np.array([2, 2, 2])  # 2mm voxels
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    # Set up source and detector parameters
    geo.DSD = 1536  # Distance Source-Detector (mm)
    geo.DSO = 1000  # Distance Source-Origin (mm)
    geo.nDetector = np.array([512, 512])
    geo.dDetector = np.array([0.8, 0.8])  # 0.8mm detector pixels
    geo.sDetector = geo.nDetector * geo.dDetector
    
    print(f"Geometry: {geo.nVoxel} voxels, {geo.nDetector} detector pixels")
    
    # Create phantom - simple geometric shapes
    phantom = np.zeros(geo.nVoxel, dtype=np.float32)
    
    # Add spherical objects
    center = geo.nVoxel // 2
    y, x, z = np.ogrid[:geo.nVoxel[0], :geo.nVoxel[1], :geo.nVoxel[2]]
    
    # Large sphere
    sphere1 = ((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2) < (30)**2
    phantom[sphere1] = 1.0
    
    # Small dense sphere
    sphere2 = ((x - center[1] + 20)**2 + (y - center[0] - 15)**2 + (z - center[2])**2) < (10)**2
    phantom[sphere2] = 2.0
    
    # Define angles for circular scan
    angles = np.linspace(0, 2*np.pi, 360, dtype=np.float32)
    print(f"Scan: {len(angles)} projections over 360°")
    
    # Forward projection (creates synthetic projections)
    print("Computing forward projections on Intel GPU...")
    start_time = time.time()
    projections = tigre.Ax(phantom, geo, angles)
    projection_time = time.time() - start_time
    print(f"Forward projection time: {projection_time:.2f} seconds")
    
    # Add realistic noise
    noise_level = 0.05
    projections += np.random.normal(0, noise_level * np.mean(projections), projections.shape).astype(np.float32)
    
    # FDK reconstruction
    print("Performing FDK reconstruction on Intel GPU...")
    start_time = time.time()
    reconstructed = algs.fdk(projections, geo, angles)
    reconstruction_time = time.time() - start_time
    print(f"FDK reconstruction time: {reconstruction_time:.2f} seconds")
    
    # Display results
    print(f"Reconstruction shape: {reconstructed.shape}")
    print(f"Value range: [{np.min(reconstructed):.3f}, {np.max(reconstructed):.3f}]")
    
    # Plot central slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original phantom slices
    mid = geo.nVoxel[2] // 2
    axes[0, 0].imshow(phantom[:, :, mid], cmap='gray')
    axes[0, 0].set_title('Original Phantom (Axial)')
    axes[0, 1].imshow(phantom[:, mid, :], cmap='gray')
    axes[0, 1].set_title('Original Phantom (Coronal)')
    axes[0, 2].imshow(phantom[mid, :, :], cmap='gray')
    axes[0, 2].set_title('Original Phantom (Sagittal)')
    
    # Reconstructed slices
    axes[1, 0].imshow(reconstructed[:, :, mid], cmap='gray')
    axes[1, 0].set_title('FDK Reconstruction (Axial)')
    axes[1, 1].imshow(reconstructed[:, mid, :], cmap='gray')
    axes[1, 1].set_title('FDK Reconstruction (Coronal)')
    axes[1, 2].imshow(reconstructed[mid, :, :], cmap='gray')
    axes[1, 2].set_title('FDK Reconstruction (Sagittal)')
    
    plt.tight_layout()
    plt.savefig('sycl_example_1_fdk.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Example 1 completed successfully!")
    return phantom, projections, reconstructed


def example_2_iterative_cgls():
    """
    Example 2: Iterative CGLS reconstruction with memory optimization
    
    This example shows how to use iterative algorithms with Intel GPU
    acceleration and memory optimization for better performance.
    """
    print("=" * 60)
    print("Example 2: Iterative CGLS with Memory Optimization")
    print("=" * 60)
    
    # Configure SYCL backend with memory optimization
    tigre.configure_backend('sycl')
    tigre.set_gpu_memory_policy('conservative')
    
    # Create smaller geometry for iterative reconstruction
    geo = tigre.geometry(mode='cone', nVoxel=np.array([96, 96, 96]))
    geo.dVoxel = np.array([2.5, 2.5, 2.5])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = 1200
    geo.DSO = 800
    geo.nDetector = np.array([384, 384])
    geo.dDetector = np.array([1.0, 1.0])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    print(f"Optimized geometry: {geo.nVoxel} voxels")
    
    # Create more complex phantom
    phantom = np.zeros(geo.nVoxel, dtype=np.float32)
    
    # Background
    phantom[:, :, :] = 0.1
    
    # Multiple objects with different densities
    center = geo.nVoxel // 2
    y, x, z = np.ogrid[:geo.nVoxel[0], :geo.nVoxel[1], :geo.nVoxel[2]]
    
    # Objects simulating different tissues
    # Bone-like structure
    bone = ((x - center[1])**2 + (y - center[0])**2) < (25)**2
    bone_mask = np.logical_and(bone, np.abs(z - center[2]) < 30)
    phantom[bone_mask] = 1.5
    
    # Soft tissue
    tissue = ((x - center[1] + 15)**2 + (y - center[0] - 10)**2 + (z - center[2] + 10)**2) < (20)**2
    phantom[tissue] = 0.8
    
    # High contrast feature
    contrast = ((x - center[1] - 20)**2 + (y - center[0] + 15)**2 + (z - center[2])**2) < (8)**2
    phantom[contrast] = 2.2
    
    # Sparse view acquisition (undersampled)
    angles = np.linspace(0, 2*np.pi, 120, dtype=np.float32)  # Reduced angles
    print(f"Undersampled scan: {len(angles)} projections")
    
    # Forward projection
    print("Computing projections...")
    projections = tigre.Ax(phantom, geo, angles)
    
    # Add noise
    projections += np.random.normal(0, 0.03 * np.mean(projections), projections.shape).astype(np.float32)
    
    # CGLS iterative reconstruction
    print("Performing CGLS iterative reconstruction...")
    
    # Set CGLS parameters
    niter = 20
    
    start_time = time.time()
    reconstructed = algs.cgls(projections, geo, angles, niter=niter, verbose=True)
    reconstruction_time = time.time() - start_time
    
    print(f"CGLS reconstruction ({niter} iterations): {reconstruction_time:.2f} seconds")
    
    # Compare with FDK for reference
    print("Computing FDK for comparison...")
    fdk_result = algs.fdk(projections, geo, angles)
    
    # Calculate metrics
    mse_cgls = np.mean((reconstructed - phantom)**2)
    mse_fdk = np.mean((fdk_result - phantom)**2)
    
    print(f"MSE - CGLS: {mse_cgls:.6f}, FDK: {mse_fdk:.6f}")
    print(f"CGLS improvement: {(mse_fdk/mse_cgls):.2f}x better")
    
    # Display comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    mid = geo.nVoxel[2] // 2
    vmin, vmax = 0, 2.5
    
    # Original and reconstructions
    axes[0, 0].imshow(phantom[:, :, mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Original Phantom')
    axes[0, 1].imshow(fdk_result[:, :, mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('FDK Reconstruction')
    axes[0, 2].imshow(reconstructed[:, :, mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('CGLS Reconstruction')
    
    # Difference images
    axes[0, 3].imshow(np.abs(reconstructed[:, :, mid] - phantom[:, :, mid]), cmap='hot')
    axes[0, 3].set_title('CGLS Error')
    
    # Profiles for quantitative comparison
    profile_line = geo.nVoxel[0] // 2
    x_coords = np.arange(geo.nVoxel[1])
    
    axes[1, 0].plot(x_coords, phantom[profile_line, :, mid], 'k-', label='Original', linewidth=2)
    axes[1, 0].plot(x_coords, fdk_result[profile_line, :, mid], 'r--', label='FDK', linewidth=2)
    axes[1, 0].plot(x_coords, reconstructed[profile_line, :, mid], 'b:', label='CGLS', linewidth=2)
    axes[1, 0].set_title('Central Profile Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence plot (if available)
    axes[1, 1].text(0.5, 0.5, f'CGLS MSE: {mse_cgls:.6f}\nFDK MSE: {mse_fdk:.6f}\n\nImprovement: {(mse_fdk/mse_cgls):.2f}x', 
                   transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Reconstruction Metrics')
    axes[1, 1].axis('off')
    
    # Memory usage info
    memory_info = tigre.get_gpu_memory_info()
    axes[1, 2].text(0.5, 0.5, f'GPU Memory Usage:\n\nAllocated: {memory_info.get("allocated", "N/A")}\nCached: {memory_info.get("cached", "N/A")}\nReserved: {memory_info.get("reserved", "N/A")}',
                   transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 2].set_title('Memory Information')
    axes[1, 2].axis('off')
    
    # Remove empty subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('sycl_example_2_cgls.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Example 2 completed successfully!")
    return phantom, reconstructed, fdk_result


def example_3_backend_comparison():
    """
    Example 3: Multi-backend performance comparison
    
    This example compares performance across different backends (SYCL, CUDA, CPU)
    for the same reconstruction task.
    """
    print("=" * 60)
    print("Example 3: Multi-Backend Performance Comparison")
    print("=" * 60)
    
    # Get available backends
    available_backends = tigre.list_backends()
    print(f"Available backends: {available_backends}")
    
    # Create test geometry
    geo = tigre.geometry(mode='cone', nVoxel=np.array([64, 64, 64]))
    geo.dVoxel = np.array([3, 3, 3])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = 1000
    geo.DSO = 600
    geo.nDetector = np.array([256, 256])
    geo.dDetector = np.array([1.5, 1.5])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    # Create test phantom
    phantom = np.zeros(geo.nVoxel, dtype=np.float32)
    center = geo.nVoxel // 2
    y, x, z = np.ogrid[:geo.nVoxel[0], :geo.nVoxel[1], :geo.nVoxel[2]]
    sphere = ((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2) < (20)**2
    phantom[sphere] = 1.5
    
    angles = np.linspace(0, 2*np.pi, 180, dtype=np.float32)
    
    # Benchmark results
    benchmark_results = {}
    
    # Test each available backend
    backends_to_test = ['sycl', 'cuda', 'cpu']
    backends_to_test = [b for b in backends_to_test if b in available_backends]
    
    for backend in backends_to_test:
        print(f"\nTesting {backend.upper()} backend...")
        
        try:
            # Configure backend
            tigre.configure_backend(backend)
            
            # Warm up GPU if applicable
            if backend in ['sycl', 'cuda']:
                _ = tigre.Ax(phantom[:32, :32, :32], geo, angles[:10])
            
            # Benchmark forward projection
            start_time = time.time()
            projections = tigre.Ax(phantom, geo, angles)
            forward_time = time.time() - start_time
            
            # Benchmark FDK reconstruction
            start_time = time.time()
            reconstructed = algs.fdk(projections, geo, angles)
            reconstruction_time = time.time() - start_time
            
            # Store results
            benchmark_results[backend] = {
                'forward_time': forward_time,
                'reconstruction_time': reconstruction_time,
                'total_time': forward_time + reconstruction_time,
                'success': True,
                'error': None
            }
            
            print(f"  Forward projection: {forward_time:.3f}s")
            print(f"  FDK reconstruction: {reconstruction_time:.3f}s")
            print(f"  Total time: {forward_time + reconstruction_time:.3f}s")
            
        except Exception as e:
            print(f"  Error with {backend}: {str(e)}")
            benchmark_results[backend] = {
                'forward_time': float('inf'),
                'reconstruction_time': float('inf'), 
                'total_time': float('inf'),
                'success': False,
                'error': str(e)
            }
    
    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    if len([r for r in benchmark_results.values() if r['success']]) > 0:
        # Find baseline (CPU or slowest)
        baseline_backend = 'cpu' if 'cpu' in benchmark_results and benchmark_results['cpu']['success'] else \
                          min(benchmark_results.keys(), key=lambda x: benchmark_results[x]['total_time'] if benchmark_results[x]['success'] else float('inf'))
        
        baseline_time = benchmark_results[baseline_backend]['total_time']
        
        print(f"{'Backend':<10} {'Forward':<12} {'Recon':<12} {'Total':<12} {'Speedup':<10} {'Status'}")
        print("-" * 70)
        
        for backend, results in benchmark_results.items():
            if results['success']:
                speedup = baseline_time / results['total_time']
                print(f"{backend.upper():<10} {results['forward_time']:<12.3f} {results['reconstruction_time']:<12.3f} "
                      f"{results['total_time']:<12.3f} {speedup:<10.2f}x {'✓'}")
            else:
                print(f"{backend.upper():<10} {'---':<12} {'---':<12} {'---':<12} {'---':<10} {'✗ ' + results['error'][:20]}")
    
    # Create visualization
    successful_backends = [b for b, r in benchmark_results.items() if r['success']]
    
    if len(successful_backends) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison bar chart
        backend_names = [b.upper() for b in successful_backends]
        forward_times = [benchmark_results[b]['forward_time'] for b in successful_backends]
        recon_times = [benchmark_results[b]['reconstruction_time'] for b in successful_backends]
        
        x = np.arange(len(backend_names))
        width = 0.35
        
        ax1.bar(x - width/2, forward_times, width, label='Forward Projection', alpha=0.8)
        ax1.bar(x + width/2, recon_times, width, label='FDK Reconstruction', alpha=0.8)
        
        ax1.set_xlabel('Backend')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Performance Comparison by Operation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(backend_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup comparison
        total_times = [benchmark_results[b]['total_time'] for b in successful_backends]
        max_time = max(total_times)
        speedups = [max_time / t for t in total_times]
        
        colors = ['red' if b == 'cpu' else 'blue' if b == 'sycl' else 'green' for b in successful_backends]
        bars = ax2.bar(backend_names, speedups, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Backend')
        ax2.set_ylabel('Relative Speedup')
        ax2.set_title('Speedup Relative to Slowest Backend')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speedup:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('sycl_example_3_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\nExample 3 completed successfully!")
    return benchmark_results


def example_4_batch_processing():
    """
    Example 4: Large dataset processing with batch optimization
    
    This example demonstrates how to efficiently process large datasets
    using batch processing and memory optimization.
    """
    print("=" * 60)
    print("Example 4: Batch Processing for Large Datasets")
    print("=" * 60)
    
    # Configure SYCL with aggressive memory policy for better throughput
    tigre.configure_backend('sycl')
    tigre.set_gpu_memory_policy('aggressive')
    
    # Create geometry for larger problem
    geo = tigre.geometry(mode='cone', nVoxel=np.array([200, 200, 200]))
    geo.dVoxel = np.array([1, 1, 1])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = 2000
    geo.DSO = 1500
    geo.nDetector = np.array([800, 600])  # Realistic detector size
    geo.dDetector = np.array([0.5, 0.5])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    print(f"Large geometry: {geo.nVoxel} voxels, {geo.nDetector} detector")
    print(f"Memory requirements: ~{np.prod(geo.nVoxel) * 4 / 1024**3:.1f} GB for volume")
    
    # Create complex phantom
    phantom = np.zeros(geo.nVoxel, dtype=np.float32)
    
    # Add multiple structures at different scales
    center = geo.nVoxel // 2
    y, x, z = np.ogrid[:geo.nVoxel[0], :geo.nVoxel[1], :geo.nVoxel[2]]
    
    # Large background structure
    ellipsoid = ((x - center[1])/80)**2 + ((y - center[0])/60)**2 + ((z - center[2])/100)**2 < 1
    phantom[ellipsoid] = 0.5
    
    # Multiple smaller high-contrast features
    np.random.seed(42)  # Reproducible results
    for i in range(20):
        cx = center[1] + np.random.randint(-60, 60)
        cy = center[0] + np.random.randint(-40, 40)
        cz = center[2] + np.random.randint(-80, 80)
        radius = np.random.randint(5, 15)
        density = np.random.uniform(1.0, 2.5)
        
        sphere = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) < radius**2
        phantom[sphere] = density
    
    # Large number of projections
    angles = np.linspace(0, 4*np.pi, 1800, dtype=np.float32)  # 1800 projections
    print(f"Processing {len(angles)} projections")
    
    # Determine optimal batch size
    optimal_batch = tigre.get_optimal_batch_size(geo, device='sycl')
    print(f"Optimal batch size: {optimal_batch} projections")
    
    # Process in batches
    print("Starting batch processing...")
    
    all_projections = []
    total_forward_time = 0
    
    for i in range(0, len(angles), optimal_batch):
        batch_end = min(i + optimal_batch, len(angles))
        batch_angles = angles[i:batch_end]
        batch_size = len(batch_angles)
        
        print(f"  Processing batch {i//optimal_batch + 1}/{(len(angles) + optimal_batch - 1)//optimal_batch}: "
              f"projections {i+1}-{batch_end} ({batch_size} projections)")
        
        # Forward projection for batch
        start_time = time.time()
        batch_projections = tigre.Ax(phantom, geo, batch_angles)
        batch_time = time.time() - start_time
        total_forward_time += batch_time
        
        all_projections.append(batch_projections)
        
        print(f"    Batch time: {batch_time:.2f}s ({batch_time/batch_size:.3f}s per projection)")
        
        # Memory usage monitoring
        memory_info = tigre.get_gpu_memory_info()
        if 'allocated' in memory_info:
            print(f"    GPU memory: {memory_info['allocated']}")
    
    # Combine all projections
    print("Combining batch results...")
    projections = np.concatenate(all_projections, axis=2)
    
    print(f"Total forward projection time: {total_forward_time:.2f}s")
    print(f"Average per projection: {total_forward_time/len(angles):.4f}s")
    print(f"Projection data size: {projections.nbytes / 1024**3:.2f} GB")
    
    # Add noise
    projections += np.random.normal(0, 0.02 * np.mean(projections), projections.shape).astype(np.float32)
    
    # Batch reconstruction with subset of angles for demonstration
    print("\nPerforming reconstruction with angle subsampling...")
    
    # Use every 5th angle for faster reconstruction
    subsample_factor = 5
    subsampled_angles = angles[::subsample_factor]
    subsampled_projections = projections[:, :, ::subsample_factor]
    
    print(f"Reconstruction using {len(subsampled_angles)} subsampled projections")
    
    # FDK reconstruction
    start_time = time.time()
    reconstructed = algs.fdk(subsampled_projections, geo, subsampled_angles)
    reconstruction_time = time.time() - start_time
    
    print(f"FDK reconstruction time: {reconstruction_time:.2f}s")
    
    # Quality metrics
    print("\nAssessing reconstruction quality...")
    
    # Central region of interest for evaluation
    roi_slice = slice(50, 150)  # Central 100x100x100 region
    phantom_roi = phantom[roi_slice, roi_slice, roi_slice]
    recon_roi = reconstructed[roi_slice, roi_slice, roi_slice]
    
    mse = np.mean((recon_roi - phantom_roi)**2)
    psnr = 20 * np.log10(np.max(phantom_roi) / np.sqrt(mse))
    
    print(f"MSE (ROI): {mse:.6f}")
    print(f"PSNR (ROI): {psnr:.2f} dB")
    
    # Performance summary
    total_processing_time = total_forward_time + reconstruction_time
    throughput = len(angles) / total_processing_time
    
    print(f"\nPerformance Summary:")
    print(f"Total processing time: {total_processing_time:.2f}s")
    print(f"Projection throughput: {throughput:.2f} projections/second")
    print(f"Data throughput: {projections.nbytes / total_processing_time / 1024**3:.2f} GB/s")
    
    # Visualization of select slices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Select representative slices
    slices_to_show = [geo.nVoxel[2]//4, geo.nVoxel[2]//2, 3*geo.nVoxel[2]//4]
    
    for i, slice_idx in enumerate(slices_to_show):
        axes[0, i].imshow(phantom[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'Original Phantom (z={slice_idx})')
        
        axes[1, i].imshow(reconstructed[:, :, slice_idx], cmap='gray')
        axes[1, i].set_title(f'Reconstruction (z={slice_idx})')
    
    plt.tight_layout()
    plt.savefig('sycl_example_4_batch.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Example 4 completed successfully!")
    return phantom, projections, reconstructed


def run_all_examples():
    """
    Run all SYCL backend examples sequentially.
    """
    print("TIGRE SYCL Backend - Complete Example Suite")
    print("=" * 60)
    
    # Check SYCL availability
    try:
        tigre.configure_backend('sycl')
        print("✅ SYCL backend available")
    except Exception as e:
        print(f"❌ SYCL backend not available: {e}")
        return
    
    examples = [
        ("Basic FDK Reconstruction", example_1_basic_fdk_sycl),
        ("Iterative CGLS Reconstruction", example_2_iterative_cgls),
        ("Backend Performance Comparison", example_3_backend_comparison),
        ("Batch Processing", example_4_batch_processing)
    ]
    
    results = {}
    
    for name, example_func in examples:
        print(f"\n{'='*20} Starting: {name} {'='*20}")
        try:
            start_time = time.time()
            result = example_func()
            execution_time = time.time() - start_time
            
            results[name] = {
                'success': True,
                'result': result,
                'time': execution_time,
                'error': None
            }
            print(f"✅ {name} completed in {execution_time:.2f}s")
            
        except Exception as e:
            results[name] = {
                'success': False,
                'result': None,
                'time': 0,
                'error': str(e)
            }
            print(f"❌ {name} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total_time = sum(r['time'] for r in results.values())
    
    print(f"Completed: {successful}/{len(examples)} examples")
    print(f"Total execution time: {total_time:.2f}s")
    
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.2f}s" if result['success'] else "N/A"
        error_str = f" - {result['error']}" if result['error'] else ""
        print(f"{status} {name}: {time_str}{error_str}")
    
    return results


if __name__ == "__main__":
    # Run individual examples or all together
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_fdk_sycl()
        elif example_num == "2":
            example_2_iterative_cgls()
        elif example_num == "3":
            example_3_backend_comparison()
        elif example_num == "4":
            example_4_batch_processing()
        else:
            print("Usage: python sycl_examples.py [1|2|3|4]")
    else:
        # Run all examples
        run_all_examples()