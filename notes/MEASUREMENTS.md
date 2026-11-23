# SYCL backend: measurements on Intel GPUs and CPUs

These measurements were made with builds of the SYCL backend on TIGRE 3.2.0 (branch
`sycl-on-3.2.0`; the commits are named below), before it was moved to TIGRE 3.3.0.

Upstream TIGRE runs only on NVIDIA GPUs through CUDA and has no CPU fallback. With the SYCL
backend in this branch, the same Python code runs on Intel GPUs through Level Zero and on Intel CPUs
through the Intel OpenCL CPU runtime. Tested hardware: an Intel Arc 140T integrated GPU, Intel
Arc Pro B50 and Arc Pro B70 discrete GPUs, and the CPUs of the two machines below. The Arc Pro B70
replaced the B50 in the desktop; see [Arc Pro B70](#arc-pro-b70) at the end.

**Hardware and software:**
- **Laptop:** Intel Core Ultra 9 285H, with an Intel Arc 140T integrated GPU (GPU driver 32.0.101.9033, Level Zero 1.17.39500+2); 64 GB RAM.
- **Desktop:** Intel Core Ultra 7 265F, with an Intel Arc Pro B50 discrete GPU, 16 GB (GPU driver 32.0.101.8976, Level Zero 1.15.39183+4); 64 GB RAM.
- **Both:** Windows 11 Pro 26H2 (build 26300); Intel OpenCL CPU runtime 2025.2.0 (driver 2025.20.6.0.04); oneAPI DPC++/C++ 2025.2.0 (`icx`, conda-forge); Python 3.12.16, NumPy 2.2.6; a build of commit e1fab7e on the laptop and of commit 54e0ad1 on the desktop (54e0ad1 changes only the tvdenoise exchange between two or more devices, which these single-device runs do not use).

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
| Small | Ax | 0.00153 (0.00102–0.00173) | 0.00226 (0.00187–0.0027) | 0.000997 (0.000915–0.00122) | 0.00302 (0.0024–0.0033) |
| Small | Atb | 0.00121 (0.00119–0.0017) | 0.00222 (0.00193–0.00267) | 0.0011 (0.00105–0.00114) | 0.00184 (0.00145–0.0029) |
| Small | FDK | 0.0035 (0.00332–0.00628) | 0.0038 (0.00366–0.00541) | 0.00296 (0.00294–0.00307) | 0.00501 (0.00448–0.00534) |
| Small | minTV, 10 it. | 0.00266 (0.00208–0.00303) | 0.00303 (0.00241–0.00382) | 0.000953 (0.000947–0.00102) | 0.00422 (0.00338–0.00553) |
| 128³ | Ax | 0.0165 (0.0142–0.0177) | 0.107 (0.101–0.122) | 0.01 (0.00928–0.0113) | 0.0585 (0.058–0.0594) |
| 128³ | Atb | 0.022 (0.0214–0.0234) | 0.125 (0.119–0.152) | 0.0127 (0.0124–0.0128) | 0.0787 (0.0783–0.0858) |
| 128³ | FDK | 0.035 (0.0322–0.0377) | 0.133 (0.129–0.137) | 0.023 (0.0228–0.0233) | 0.0885 (0.0877–0.0906) |
| 128³ | minTV, 10 it. | 0.0231 (0.0221–0.0272) | 0.0157 (0.0151–0.0162) | 0.00601 (0.00597–0.00613) | 0.0122 (0.012–0.0132) |
| Large, 4.25 GiB | Ax | 2.23 (2.13–2.33) | 3.66 (3.57–3.72) | 1.6 (1.58–1.61) | 2.93 (2.87–2.97) |
| Large, 4.25 GiB | Atb | 2.62 (2.47–2.7) | 8.48 (8.22–8.6) | 1.48 (1.48–1.55) | 5.96 (5.86–6.16) |
| Large, 4.25 GiB | FDK | 2.66 (2.6–2.69) | 8.9 (8.74–8.94) | 1.53 (1.53–1.55) | 7.43 (7.18–7.51) |
| Large, 4.25 GiB | minTV, 10 it. | 12.1 (11.8–12.1) | 15.8 (15.8–16.2) | 3.62 (3.58–3.66) | 14.9 (14.9–15) |

First call in the process, median of runs 2–5, in seconds:

| Problem | Operation | Arc 140T | Core Ultra 9 285H | Arc Pro B50 | Core Ultra 7 265F |
|---|---|---:|---:|---:|---:|
| Small | Ax | 0.26 | 0.723 | 0.249 | 0.548 |
| Small | Atb | 0.264 | 0.741 | 0.256 | 0.585 |
| Small | FDK | 0.259 | 0.756 | 0.257 | 0.595 |
| Small | minTV, 10 it. | 0.254 | 0.739 | 0.26 | 0.592 |
| 128³ | Ax | 0.258 | 0.824 | 0.263 | 0.608 |
| 128³ | Atb | 0.26 | 0.814 | 0.271 | 0.666 |
| 128³ | FDK | 0.274 | 0.848 | 0.283 | 0.669 |
| 128³ | minTV, 10 it. | 0.261 | 0.736 | 0.265 | 0.6 |
| Large, 4.25 GiB | Ax | 2.42 | 4.3 | 1.88 | 3.36 |
| Large, 4.25 GiB | Atb | 2.82 | 8.94 | 1.78 | 5.47 |
| Large, 4.25 GiB | FDK | 2.85 | 8.88 | 1.76 | 6 |
| Large, 4.25 GiB | minTV, 10 it. | 12 | 16.3 | 3.83 | 14.4 |

**Parity, GPU vs CPU device on the same machine and the same inputs** (values are relative to the largest value in the output; large-volume Atb, FDK and minTV images are compared on a 1/64 subsample):
- **Ax:** max. difference ≤ 2.4e-5 and relative L2 ≤ 2.3e-6 at all sizes on both machines; no ray differs by more than 1e-3.
- **Atb and FDK:** max. difference ≤ 4.2e-5 and relative L2 ≤ 9.3e-6 at all sizes on both machines.
- **minTV, small and 128³:** max. difference ≤ 1.2e-3 and relative L2 ≤ 6.9e-6. Both devices are as close to a float64 NumPy reference as a float32 NumPy run is.
- **minTV, large volume:** without splitting (Arc Pro B50 and both CPU devices), the results agree within 7.8e-5 max. difference and 1.7e-6 relative L2. The split run on the Arc 140T differs from them by up to 1.6e-3 (relative L2 3.0e-4).

**Changes from the previous builds** (51e8dfd on the laptop, 538033f on the desktop), same method, inputs and software:
- **Ax (Siddon), from the port of #865:** on all four devices, Atb, FDK and minTV outputs are bitwise identical to the previous builds; only Ax changed. Before, a few rays crossed different voxels on the GPU and the CPU device (GPU vs CPU max. difference 5.7e-2 at 128³ and 1.3e-2 on the large volume); now the largest GPU vs CPU difference is 2.4e-5. The Ax outputs moved by up to 5.7e-2 (128³) and 2.6e-2 (large volume) of their peak value, on a few rays.
- **Ax split into pieces:** a 256³ random image projected with a 24 MB limit (`TIGRE_SYCL_MEMORY_LIMIT_MB=24`) differed from the unsplit projection by up to 1.6e-1 before and by 9.0e-6 now, on all four devices.
- **Ax timings,** paired runs of both builds in the same task (second call, median of runs 2–5, in seconds; before → after):

| Problem | Arc 140T | Core Ultra 9 285H | Arc Pro B50 | Core Ultra 7 265F |
|---|---:|---:|---:|---:|
| Small | 0.00125 → 0.00111 | 0.00225 → 0.00212 | 0.00091 → 0.00102 | 0.00405 → 0.00338 |
| 128³ | 0.0240 → 0.0157 | 0.108 → 0.109 | 0.0137 → 0.0103 | 0.0615 → 0.0593 |
| Large, 4.25 GiB | 2.41 → 2.46 | 3.50 → 3.72 | 1.64 → 1.66 | 2.81 → 2.96 |
| 256³ random image, cone beam, 64 angles | 0.117 → 0.087 | 0.490 → 0.484 | 0.0613 → 0.0479 | 0.278 → 0.278 |
| 256³ random image, parallel beam, 64 angles | 0.105 → 0.101 | 0.440 → 0.441 | 0.0578 → 0.0512 | 0.254 → 0.256 |

- **tvdenoise split into pieces, from the port of #872 and #873:** with 24, 12, 8 and 6 MB limits (a 64×256×256 random image, 10 and 100 iterations; the 128³ head phantom, 50 iterations), the split result is now bitwise equal to the unsplit one on all four devices; before, it differed by up to 23% of the peak value. With the Arc Pro B50 and the Core Ultra 7 265F in one process (`TIGRE_SYCL_ALL_DEVICES=1`), split and unsplit agree within 6.7e-7. The unsplit result changed in the first and last few slices, as in the CUDA build. minTV and AwminTV are unchanged: each piece is still normalised by its own gradient norm, so split results differ from unsplit ones by up to 2.1e-2 (minTV, 128³ head phantom, 8 MB). With a 6 MB limit, minTV ended the Python process on both machines, with both builds; this has not been looked into yet.
- **Other second-call medians** moved by more than 5% and outside the previous run-to-run range (previous → this build, Ax excluded, see above): Core Ultra 9 285H small minTV 0.0024 → 0.00303 s; Arc Pro B50 128³ FDK 0.0244 → 0.023 s; Core Ultra 7 265F small Atb 0.00299 → 0.00184 s, small minTV 0.00247 → 0.00422 s and large Atb 6.3 → 5.96 s. The code of these operations did not change. All other second-call medians are within 5% of the previous figures or inside their run-to-run range.

## Arc Pro B70

The Arc Pro B50 in the desktop was replaced by an Intel Arc Pro B70, 32 GB (GPU driver 32.0.101.8976,
Level Zero 1.15.39183+4). It reports 31906 MiB, of which TIGRE plans with 30311 MiB, so the large image
is not split.
Same machine, software, build (54e0ad1), inputs and method as above; the Core Ultra 7 265F was
measured again in the same runs.

Second call, median (min–max) of runs 2–5, in seconds; the Arc Pro B50 column is repeated from above:

| Problem | Operation | Arc Pro B70 | Arc Pro B50 | B70/B50 |
|---|---|---:|---:|---:|
| Small | Ax | 0.000985 (0.000891–0.00101) | 0.000997 | 0.99 |
| Small | Atb | 0.000971 (0.000965–0.00111) | 0.0011 | 0.88 |
| Small | FDK | 0.00305 (0.00285–0.00349) | 0.00296 | 1.03 |
| Small | minTV, 10 it. | 0.000814 (0.000804–0.000823) | 0.000953 | 0.85 |
| 128³ | Ax | 0.00667 (0.00662–0.00683) | 0.01 | 0.66 |
| 128³ | Atb | 0.00824 (0.00795–0.00829) | 0.0127 | 0.65 |
| 128³ | FDK | 0.019 (0.0183–0.0195) | 0.023 | 0.83 |
| 128³ | minTV, 10 it. | 0.00427 (0.00425–0.00431) | 0.00601 | 0.71 |
| Large, 4.25 GiB | Ax | 1.61 (1.52–1.63) | 1.6 | 1.01 |
| Large, 4.25 GiB | Atb | 1.22 (1.2–1.23) | 1.48 | 0.82 |
| Large, 4.25 GiB | FDK | 1.25 (1.21–1.28) | 1.53 | 0.81 |
| Large, 4.25 GiB | minTV, 10 it. | 2.37 (2.28–2.4) | 3.62 | 0.66 |

- **First call:** B70/B50 0.94–1.00 for the small and 128³ problems (about 0.25 s, mostly start-up);
  on the large volume Ax 1.84 s (0.98), Atb 1.47 s (0.83), FDK 1.48 s (0.84), minTV 2.61 s (0.68).
- **Siddon Ax, 256³ random image, 64 angles:** cone beam 0.0341 s (B70/B50 0.71), parallel beam
  0.0356 s (0.70).
- **tvdenoise:** 128³ head phantom, 50 iterations, 0.0102 s; 64×256×256 random image, 100 iterations,
  0.0473 s (no B50 figures).
- **Core Ultra 7 265F** in the same runs: within 5% of the figures above at 128³ and on the large
  volume; the small-problem times vary more from run to run (e.g. small FDK 0.00425 s).
- **Parity:** Arc Pro B70 vs CPU device gives the same figures as the Arc Pro B50 vs CPU device above.
  The B70 outputs are bitwise equal to the B50 outputs, except minTV (up to 1.8e-6 of the maximum at
  128³ and 5.7e-5 on the large volume).

## HyperX, 2x B50, Windows

One and two Intel Arc Pro B50 GPUs (16 GB each) in the same desktop, with the SYCL backend on TIGRE
3.3.0. These are baseline numbers on 3.3.0; TIGRE 3.3.1 reworks how multi-GPU Ax and Atb split the work.

**Hardware and software:** Intel Core Ultra 7 265F, 64 GB RAM, two Intel Arc Pro B50 (GPU driver
32.0.101.8976, Level Zero 1.15.39183+4; no other GPU installed); Windows 11 Pro 26H2 (build 26300);
oneAPI DPC++/C++ 2025.2.0 (`icx`, conda-forge); Python 3.12.16, NumPy 2.2.6; a build of branch
`sycl-on-3.3.0`, commit e736348. `sycl-ls` and TIGRE both list the two B50s.

**Method:** cone beam, TIGRE head phantom, n³ voxels with the default geometry, n×n detector
(pixel 0.8 mm × 512/n), 360 angles over 360°, for n = 256 and n = 512. The projections are made once
with `tigre.Ax` on one GPU and used for all the other operations. Each operation runs in one Python
process, started from a one-shot task in the Windows desktop session: first with `gpuids` set to
device 0, then with devices 0 and 1. Each of the two runs is one warm-up call followed by 5 timed
calls; the table gives the median of the 5 calls. Times are wall-clock time around the Python call and
include host–device transfers. Scaling is the 1-GPU median divided by the 2-GPU median.

Operations: `tigre.Ax` (interpolated), `tigre.Atb` (FDK weights and matched),
`tigre.algorithms.fdk`, `tigre.algorithms.cgls` (10 iterations), `tigre.algorithms.ossart`
(5 iterations, block size 20), and `minTV` on the phantom image (alpha 0.01, 50 iterations).

Median (min–max) of 5 calls after one warm-up, in seconds:

| Size | Operation | 1 GPU | 2 GPUs | Scaling |
|---|---|---:|---:|---:|
| 256³ | Ax, interpolated | 1.68 (1.68–1.69) | 0.852 (0.848–0.855) | 1.98 |
| 256³ | Atb, FDK weights | 0.341 (0.34–0.342) | 0.185 (0.184–0.187) | 1.85 |
| 256³ | Atb, matched | 0.348 (0.347–0.348) | 0.183 (0.182–0.184) | 1.90 |
| 256³ | FDK | 0.419 (0.417–0.421) | 0.264 (0.261–0.295) | 1.59 |
| 256³ | CGLS, 10 it. | 10.3 (10.2–10.5) | 6.7 (6.69–6.97) | 1.54 |
| 256³ | OS-SART, 5 it. | 8.8 (8.75–9.27) | 8.56 (8.52–8.66) | 1.03 |
| 256³ | minTV, 50 it. | 0.221 (0.221–0.222) | 0.144 (0.144–0.145) | 1.54 |
| 512³ | Ax, interpolated | 14.1 (14.1–14.2) | 7.13 (7.12–7.16) | 1.98 |
| 512³ | Atb, FDK weights | 2.71 (2.71–2.72) | 1.46 (1.46–1.47) | 1.86 |
| 512³ | Atb, matched | 2.77 (2.77–2.77) | 1.43 (1.43–1.43) | 1.93 |
| 512³ | FDK | 3.01 (3–3.03) | 1.75 (1.75–1.77) | 1.72 |
| 512³ | CGLS, 10 it. | 78.8 (78.8–79.0) | 48.2 (48.1–49.3) | 1.64 |
| 512³ | OS-SART, 5 it. | 69.6 (69.5–69.6) | 85.5 (83.2–85.8) | 0.81 |
| 512³ | minTV, 50 it. | 1.62 (1.62–1.62) | 1.02 (1.02–1.02) | 1.59 |

- **1 GPU vs 2 GPUs:** for every operation and both sizes, the 2-GPU output is bitwise equal to the
  1-GPU output (RMSE 0, max. difference 0); all outputs are finite.
- **512³:** all operations ran on one and on two B50s; no allocation failed.
- **OS-SART:** scaling 1.03 at 256³; at 512³ it took longer on two GPUs than on one (85.5 s vs
  69.6 s, scaling 0.81).
- **Warm-up call** (in-process start-up included): at most 0.61 s longer than the timed median
  (256³ CGLS and OS-SART on one GPU). The 2-GPU warm-up call of OS-SART at 512³ took 79.9 s, 5.6 s
  less than the timed median.
- In the load samples taken during the runs (about once a minute), no program other than the
  measurement process and Windows (`System`) used the GPUs.
