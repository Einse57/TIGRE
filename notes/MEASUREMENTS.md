# SYCL backend: measurements on Intel GPUs and CPUs

Upstream TIGRE runs only on NVIDIA GPUs through CUDA and has no CPU fallback. With the SYCL
backend in this branch, the same Python code runs on Intel GPUs through Level Zero and on Intel CPUs
through the Intel OpenCL CPU runtime. Tested hardware: an Intel Arc 140T integrated GPU and an Intel
Arc Pro B50 discrete GPU, and the CPUs of the two machines below.

**Hardware and software:**
- **Laptop:** Intel Core Ultra 9 285H, with an Intel Arc 140T integrated GPU (GPU driver 32.0.101.9033, Level Zero 1.17.39500+2); 64 GB RAM.
- **Desktop:** Intel Core Ultra 7 265F, with an Intel Arc Pro B50 discrete GPU, 16 GB (GPU driver 32.0.101.8976, Level Zero 1.15.39183+4); 64 GB RAM.
- **Both:** Windows 11 Pro 26H2 (build 26300); Intel OpenCL CPU runtime 2025.2.0 (driver 2025.20.6.0.04); oneAPI DPC++/C++ 2025.2.0 (`icx`, conda-forge); Python 3.12.16, NumPy 2.2.6; a build of commit 51e8dfd on the laptop and of commit 538033f on the desktop (538033f changes only the CI workflow, so the TIGRE code is the same).

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
| Small | Ax | 0.00125 (0.000954–0.00129) | 0.00258 (0.00196–0.00334) | 0.00103 (0.000897–0.00112) | 0.00309 (0.00247–0.00348) |
| Small | Atb | 0.00122 (0.00103–0.00127) | 0.00203 (0.00188–0.00281) | 0.00109 (0.00103–0.00123) | 0.00299 (0.00239–0.00333) |
| Small | FDK | 0.00367 (0.0036–0.00458) | 0.00505 (0.00359–0.00786) | 0.00305 (0.00293–0.00316) | 0.00453 (0.00392–0.00511) |
| Small | minTV, 10 it. | 0.0025 (0.00144–0.00268) | 0.0024 (0.0023–0.00269) | 0.000952 (0.00094–0.00103) | 0.00247 (0.0022–0.0042) |
| 128³ | Ax | 0.0232 (0.0229–0.0247) | 0.111 (0.11–0.115) | 0.0137 (0.0135–0.014) | 0.0632 (0.0625–0.0642) |
| 128³ | Atb | 0.0209 (0.0201–0.0232) | 0.132 (0.119–0.163) | 0.0126 (0.0126–0.013) | 0.0777 (0.0752–0.0808) |
| 128³ | FDK | 0.0363 (0.0355–0.0368) | 0.138 (0.136–0.143) | 0.0244 (0.0236–0.0246) | 0.0879 (0.0866–0.0892) |
| 128³ | minTV, 10 it. | 0.023 (0.0223–0.0236) | 0.0154 (0.0152–0.0161) | 0.00613 (0.00591–0.00633) | 0.0119 (0.0117–0.0128) |
| Large, 4.25 GiB | Ax | 2.31 (2.18–2.48) | 3.54 (3.48–3.62) | 1.62 (1.59–1.68) | 2.9 (2.87–2.92) |
| Large, 4.25 GiB | Atb | 2.55 (2.54–2.63) | 8.91 (8.87–9.07) | 1.52 (1.5–1.56) | 6.3 (6.15–6.49) |
| Large, 4.25 GiB | FDK | 2.71 (2.63–2.78) | 8.98 (8.72–9.13) | 1.54 (1.49–1.56) | 7.52 (7.44–7.72) |
| Large, 4.25 GiB | minTV, 10 it. | 11.9 (11.8–12) | 15.9 (15.9–16.3) | 3.62 (3.6–3.7) | 15.3 (15.2–16.4) |

First call in the process, median of runs 2–5, in seconds:

| Problem | Operation | Arc 140T | Core Ultra 9 285H | Arc Pro B50 | Core Ultra 7 265F |
|---|---|---:|---:|---:|---:|
| Small | Ax | 0.248 | 0.718 | 0.247 | 0.571 |
| Small | Atb | 0.251 | 0.741 | 0.255 | 0.605 |
| Small | FDK | 0.273 | 0.746 | 0.256 | 0.598 |
| Small | minTV, 10 it. | 0.26 | 0.721 | 0.261 | 0.588 |
| 128³ | Ax | 0.26 | 0.787 | 0.264 | 0.634 |
| 128³ | Atb | 0.259 | 0.828 | 0.271 | 0.668 |
| 128³ | FDK | 0.263 | 0.867 | 0.277 | 0.687 |
| 128³ | minTV, 10 it. | 0.272 | 0.719 | 0.265 | 0.606 |
| Large, 4.25 GiB | Ax | 2.48 | 4.01 | 1.91 | 3.35 |
| Large, 4.25 GiB | Atb | 2.76 | 9.5 | 1.78 | 5.55 |
| Large, 4.25 GiB | FDK | 2.93 | 8.8 | 1.8 | 6.08 |
| Large, 4.25 GiB | minTV, 10 it. | 12 | 16.4 | 3.83 | 14.9 |

**Parity, GPU vs CPU device on the same machine and the same inputs** (values are relative to the largest value in the output; large-volume images are compared on a 1/64 subsample):
- **Atb and FDK:** max. difference ≤ 4.2e-5 and relative L2 ≤ 9.3e-6 at all sizes on both machines.
- **Ax:** relative L2 ≤ 1.0e-4 on both machines. The largest differences (5.7e-2 at 128³, 1.3e-2 on the large volume) come from a few Siddon rays, where float32 rounding can change which voxels a ray crosses: 1 of 1.92 M rays at 128³, and 163 (Arc 140T) or 145 (Arc Pro B50) of 2.1 M on the large volume, are above 1e-3.
- **minTV, small and 128³:** max. difference ≤ 1.2e-3 and relative L2 ≤ 6.9e-6. Both devices are as close to a float64 NumPy reference as a float32 NumPy run is.
- **minTV, large volume:** without splitting (Arc Pro B50 and both CPU devices), the results agree within 8.3e-5 max. difference and 1.7e-6 relative L2. The split run on the Arc 140T differs from them by up to 1.6e-3 (relative L2 3.0e-4).

**Changes from the earlier build, commit 6b45fe1** (before TIGRE master up to #868 was merged in and its CUDA changes were ported to SYCL), same method, inputs and software:
- Outputs: all 12 problem/operation pairs are bitwise identical to the 6b45fe1 run on all four devices, so the parity figures above are unchanged.
- First call: on both GPUs, the first call on the small and 128³ problems takes about 0.08–0.16 s longer than with 6b45fe1 (for example, small Ax went from 0.108 to 0.248 s on the Arc 140T and from 0.167 to 0.247 s on the Arc Pro B50). The first call on the CPU devices is about the same. The cause has not been looked into.
- Second call, where the median moved by more than 5% and outside the earlier run-to-run range (6b45fe1 → this build):
  - Arc 140T: small FDK 0.00322 → 0.00367 s; small Atb 0.00131 → 0.00122 s; large Atb 2.7 → 2.55 s.
  - Core Ultra 9 285H: small FDK 0.00374 → 0.00505 s; small minTV 0.0026 → 0.0024 s; 128³ minTV 0.021 → 0.0154 s.
  - Arc Pro B50: small Ax 0.000922 → 0.00103 s; small minTV 0.000864 → 0.000952 s.
  - Core Ultra 7 265F: small minTV 0.00436 → 0.00247 s; 128³ minTV 0.0156 → 0.0119 s; large Atb 6.93 → 6.3 s; large minTV 14.5 → 15.3 s.
  - All other second-call medians are within 5% of the 6b45fe1 figures or inside their run-to-run range.
