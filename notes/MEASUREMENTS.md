# SYCL backend: measurements on an Intel laptop

Upstream TIGRE runs only on NVIDIA GPUs through CUDA and has no CPU fallback. With the SYCL
backend in this branch, the same Python code runs on a laptop's Intel integrated GPU and on its CPU
(through the Intel OpenCL CPU runtime). The measurements below were made on one laptop.

**Hardware and software:** Intel Core Ultra 9 285H (Intel OpenCL CPU runtime 2025.2.0, driver 2025.20.6.0.04), Intel Arc
140T integrated GPU (Level Zero 1.17.39500+2, GPU driver 32.0.101.9033), 64 GB RAM, Windows 11 Pro
26H2 (build 26300); oneAPI DPC++/C++ 2025.2.0 (`icx`, conda-forge), Python 3.12.16, NumPy 2.2.6;
build of commit 6b45fe1.

**Method:** both devices use the same build and the same inputs. Each run is a fresh Python process
started from a one-shot task in the Windows desktop session, with the two devices interleaved; there
are 5 runs per configuration, and the table gives the median (min–max) over runs 2–5, in seconds. Each
process calls the operation twice. The second call is shown as the main figure; the "first call"
columns add in-process start-up costs such as SYCL kernel JIT compilation. Times are wall-clock time
around the Python call and include host–device transfers.

Problems (cone beam, TIGRE head phantom):
- Small: 32×40×36 voxels, 56×64 detector, 20 angles.
- 128³: 128×128×128 voxels, 160×200 detector, 60 angles.
- Large: 1088×1024×1024 voxels (4.25 GiB), 512×512 detector, 8 angles. The image is larger than
  the Arc's 4095 MiB maximum single allocation, so TIGRE splits it.

Operations: `tigre.Ax` (Siddon), `tigre.Atb` (FDK weights), `tigre.algorithms.fdk`, and
`minTV` (alpha 0.5, 10 iterations). Atb and FDK use the same projections on both devices.

| Problem | Operation | Arc 140T | CPU device | Arc 140T, first call | CPU device, first call |
|---|---|---:|---:|---:|---:|
| Small | Ax | 0.00119 (0.00118–0.00163) | 0.00254 (0.00251–0.00269) | 0.108 | 0.723 |
| Small | Atb | 0.00131 (0.00124–0.00133) | 0.00211 (0.00185–0.00331) | 0.123 | 0.74 |
| Small | FDK | 0.00322 (0.00274–0.00366) | 0.00374 (0.00348–0.00478) | 0.118 | 0.67 |
| Small | minTV, 10 it. | 0.00247 (0.00132–0.00402) | 0.0026 (0.00254–0.00267) | 0.123 | 0.72 |
| 128³ | Ax | 0.023 (0.0218–0.0238) | 0.11 (0.106–0.125) | 0.129 | 0.828 |
| 128³ | Atb | 0.023 (0.0207–0.0257) | 0.121 (0.116–0.135) | 0.124 | 0.852 |
| 128³ | FDK | 0.0387 (0.0338–0.0395) | 0.142 (0.132–0.171) | 0.138 | 0.795 |
| 128³ | minTV, 10 it. | 0.022 (0.0211–0.0228) | 0.021 (0.0209–0.0216) | 0.126 | 0.707 |
| Large, 4.25 GiB | Ax | 2.4 (2.33–2.6) | 3.54 (3.42–3.73) | 2.44 | 4.04 |
| Large, 4.25 GiB | Atb | 2.7 (2.65–2.73) | 8.92 (8.81–8.98) | 2.73 | 9.59 |
| Large, 4.25 GiB | FDK | 2.6 (2.5–2.73) | 8.91 (8.76–9.15) | 2.76 | 8.88 |
| Large, 4.25 GiB | minTV, 10 it. | 11.8 (11.6–12.1) | 15.4 (15.2–15.5) | 11.7 | 15.9 |

**Parity, Arc 140T vs CPU device on the same inputs** (values are relative to the largest value in the output):
- **Atb and FDK:** max. difference ≤ 4.2e-5 and relative L2 ≤ 9.3e-6 at all three sizes, including the split large volume.
- **Ax:** relative L2 ≤ 1.0e-4. The largest differences (5.7e-2 at 128³, 1.3e-2 on the large volume) come from a few Siddon rays, where float32 rounding can change which voxels a ray crosses: 1 of 1.92 M rays at 128³ and 163 of 2.1 M on the large volume are above 1e-3.
- **minTV:** max. difference ≤ 1.6e-3 and relative L2 ≤ 3.0e-4. At the small and 128³ sizes, both devices are as close to a float64 NumPy reference as a float32 NumPy run is (max. 2.6e-3 and 1.2e-3).
