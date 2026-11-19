# SYCL backend: measurements on Intel GPUs and CPUs

Upstream TIGRE runs only on NVIDIA GPUs through CUDA and has no CPU fallback. With the SYCL
backend in this branch, the same Python code runs on Intel GPUs through Level Zero and on Intel CPUs
through the Intel OpenCL CPU runtime. Tested hardware: an Intel Arc 140T integrated GPU and an Intel
Arc Pro B50 discrete GPU, and the CPUs of the two machines below.

**Hardware and software:**
- **Laptop:** Intel Core Ultra 9 285H, with an Intel Arc 140T integrated GPU (GPU driver 32.0.101.9033, Level Zero 1.17.39500+2); 64 GB RAM.
- **Desktop:** Intel Core Ultra 7 265F, with an Intel Arc Pro B50 discrete GPU, 16 GB (GPU driver 32.0.101.8976, Level Zero 1.15.39183+4); 64 GB RAM.
- **Both:** Windows 11 Pro 26H2 (build 26300); Intel OpenCL CPU runtime 2025.2.0 (driver 2025.20.6.0.04); oneAPI DPC++/C++ 2025.2.0 (`icx`, conda-forge); Python 3.12.16, NumPy 2.2.6; a build of commit 6b45fe1 on each machine.

**Method:** on each machine, the GPU and the CPU device use the same build and the same inputs, and
the device is chosen with `ONEAPI_DEVICE_SELECTOR` (`level_zero:0` or `opencl:cpu`). Each run is a
fresh Python process started from a one-shot task in the Windows desktop session, with the two
devices interleaved; there are 5 runs per configuration. Each process calls the operation twice. The
second call is the main figure; the first call adds in-process start-up costs such as SYCL kernel JIT
compilation. Times are wall-clock time around the Python call and include host–device transfers.

Problems (cone beam, TIGRE head phantom):
- Small: 32×40×36 voxels, 56×64 detector, 20 angles.
- 128³: 128×128×128 voxels, 160×200 detector, 60 angles.
- Large: 1088×1024×1024 voxels (4.25 GiB), 512×512 detector, 8 angles. The Arc 140T can allocate at
  most 4095 MiB at once, so TIGRE splits the image there. The Arc Pro B50 can allocate 15881 MiB at
  once, so the image is not split.

Operations: `tigre.Ax` (Siddon), `tigre.Atb` (FDK weights), `tigre.algorithms.fdk`, and
`minTV` (alpha 0.5, 10 iterations). On each machine, Atb and FDK use the same projections on both devices.

Second call, median (min–max) of runs 2–5, in seconds:

| Problem | Operation | Arc 140T | Core Ultra 9 285H | Arc Pro B50 | Core Ultra 7 265F |
|---|---|---:|---:|---:|---:|
| Small | Ax | 0.00119 (0.00118–0.00163) | 0.00254 (0.00251–0.00269) | 0.000922 (0.000913–0.000927) | 0.00354 (0.00173–0.00449) |
| Small | Atb | 0.00131 (0.00124–0.00133) | 0.00211 (0.00185–0.00331) | 0.00106 (0.000957–0.00109) | 0.00236 (0.00137–0.00357) |
| Small | FDK | 0.00322 (0.00274–0.00366) | 0.00374 (0.00348–0.00478) | 0.00304 (0.00287–0.00328) | 0.00446 (0.00345–0.00575) |
| Small | minTV, 10 it. | 0.00247 (0.00132–0.00402) | 0.0026 (0.00254–0.00267) | 0.000864 (0.000861–0.000875) | 0.00436 (0.00262–0.00648) |
| 128³ | Ax | 0.023 (0.0218–0.0238) | 0.11 (0.106–0.125) | 0.0134 (0.0123–0.0139) | 0.0625 (0.0606–0.0643) |
| 128³ | Atb | 0.023 (0.0207–0.0257) | 0.121 (0.116–0.135) | 0.0127 (0.0125–0.0133) | 0.0759 (0.0708–0.0824) |
| 128³ | FDK | 0.0387 (0.0338–0.0395) | 0.142 (0.132–0.171) | 0.0234 (0.0231–0.0246) | 0.0846 (0.0831–0.0872) |
| 128³ | minTV, 10 it. | 0.022 (0.0211–0.0228) | 0.021 (0.0209–0.0216) | 0.00583 (0.00562–0.00698) | 0.0156 (0.0146–0.0158) |
| Large, 4.25 GiB | Ax | 2.4 (2.33–2.6) | 3.54 (3.42–3.73) | 1.58 (1.57–1.6) | 2.8 (2.78–2.82) |
| Large, 4.25 GiB | Atb | 2.7 (2.65–2.73) | 8.92 (8.81–8.98) | 1.49 (1.48–1.5) | 6.93 (6.81–6.95) |
| Large, 4.25 GiB | FDK | 2.6 (2.5–2.73) | 8.91 (8.76–9.15) | 1.51 (1.49–1.55) | 7.83 (7.78–7.9) |
| Large, 4.25 GiB | minTV, 10 it. | 11.8 (11.6–12.1) | 15.4 (15.2–15.5) | 3.46 (3.43–3.51) | 14.5 (14.4–14.6) |

First call in the process, median of runs 2–5, in seconds:

| Problem | Operation | Arc 140T | Core Ultra 9 285H | Arc Pro B50 | Core Ultra 7 265F |
|---|---|---:|---:|---:|---:|
| Small | Ax | 0.108 | 0.723 | 0.167 | 0.575 |
| Small | Atb | 0.123 | 0.74 | 0.171 | 0.584 |
| Small | FDK | 0.118 | 0.67 | 0.179 | 0.596 |
| Small | minTV, 10 it. | 0.123 | 0.72 | 0.174 | 0.577 |
| 128³ | Ax | 0.129 | 0.828 | 0.182 | 0.623 |
| 128³ | Atb | 0.124 | 0.852 | 0.187 | 0.652 |
| 128³ | FDK | 0.138 | 0.795 | 0.196 | 0.667 |
| 128³ | minTV, 10 it. | 0.126 | 0.707 | 0.176 | 0.592 |
| Large, 4.25 GiB | Ax | 2.44 | 4.04 | 1.8 | 3.25 |
| Large, 4.25 GiB | Atb | 2.73 | 9.59 | 1.66 | 5.88 |
| Large, 4.25 GiB | FDK | 2.76 | 8.88 | 1.68 | 6.5 |
| Large, 4.25 GiB | minTV, 10 it. | 11.7 | 15.9 | 3.57 | 14 |

**Parity, GPU vs CPU device on the same machine and the same inputs** (values are relative to the largest value in the output; large-volume images are compared on a 1/64 subsample):
- **Atb and FDK:** max. difference ≤ 4.2e-5 and relative L2 ≤ 9.3e-6 at all sizes on both machines.
- **Ax:** relative L2 ≤ 1.0e-4 on both machines. The largest differences (5.7e-2 at 128³, 1.3e-2 on the large volume) come from a few Siddon rays, where float32 rounding can change which voxels a ray crosses: 1 of 1.92 M rays at 128³, and 163 (Arc 140T) or 145 (Arc Pro B50) of 2.1 M on the large volume, are above 1e-3.
- **minTV, small and 128³:** max. difference ≤ 1.2e-3 and relative L2 ≤ 6.9e-6. Both devices are as close to a float64 NumPy reference as a float32 NumPy run is.
- **minTV, large volume:** without splitting (Arc Pro B50 and both CPU devices), the results agree within 8.3e-5 max. difference and 1.7e-6 relative L2. The split run on the Arc 140T differs from them by up to 1.6e-3 (relative L2 3.0e-4).

**Re-run on the laptop with commit 51e8dfd** (this branch with TIGRE master up to #868 merged in, and its CUDA changes ported to SYCL), same method and inputs:
- Outputs: all 12 problem/operation pairs are bitwise identical to the 6b45fe1 run, on the Arc 140T and on the CPU device. The parity figures above still hold.
- Second-call times: within the run-to-run range of the table above, except on the Core Ultra 9 285H for 128³ minTV, 0.0154 s (0.0152–0.0161); large-volume minTV, 15.9 s (15.9–16.3); and small FDK, 0.00505 s (0.00359–0.00786). Small FDK on the Arc 140T took 0.00367 s (0.0036–0.00458).
- The table above is from 6b45fe1 on both machines.
