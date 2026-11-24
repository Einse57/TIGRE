# SYCL forward projectors on 3.3.1: correctness measurements

Commit 65a65026 (SYCL Ax with the 3.3.1 page-locking rule and image splitting), built with
TIGRE_BACKEND=sycl and oneAPI DPC++ 2025.2.

## Test suites, SYCL CPU device (Linux, Intel Xeon processor, Python 3.11, NumPy 2.2.6)

| | `pytest` | `pytest -m extended` |
|---|---|---|
| 1010aeb (on 3.3.0) | 158 passed, 111 skipped | 7 passed, 20 skipped |
| 65a65026 (on 3.3.1) | 166 passed, 129 skipped | 7 passed, 21 skipped |

The difference is 3.3.1's new tests: test_thin_shapes (8 forward projection tests pass, 16 need `_Atb`),
test_backprojection_empty_gpu_piece (2, need 4 GPUs) and test_atb_int32_size (extended, needs `_Atb`).
The cherry-picked commits without the adaptation give the same counts.

Allocation retry: with the address space capped so that allocating the whole image fails while the
free-memory estimate says it fits, the four forward projectors split the image and complete (before the change
they stopped with "out of memory"). The results equal those of a split forced with TIGRE_SYCL_MEMORY_LIMIT_MB.

## Intel Arc Pro B70 (Windows, Level Zero, driver 32.0.101.8976, Python 3.12, NumPy 2.2.6)

192^3 random image, 36 angles, default geometry, against the SYCL CPU device above (same commit and inputs).
Unsplit, and split with TIGRE_SYCL_MEMORY_LIMIT_MB=16 (which page-locks the image). max|ref| is 143 to 195.

| | unsplit: max abs diff | unsplit: RMSE | split: max abs diff | split: RMSE |
|---|---|---|---|---|
| cone, Siddon | 4.4e-3 | 2.3e-4 | 4.3e-3 | 2.0e-4 |
| cone, interpolated | 2.1e-1 | 1.9e-4 | 1.3e-2 | 6.9e-5 |
| parallel, Siddon | 1.4e-1 | 4.2e-3 | 1.4e-1 | 4.2e-3 |
| parallel, interpolated | 5.9e-1 | 3.0e-2 | 5.9e-1 | 3.0e-2 |

- The B70 results of 65a65026 and 1010aeb are bit-identical, unsplit and split: the differences to the CPU device
  come from the kernels (they are concentrated in a few angles, where rays meet voxel boundaries), not from this change.
- Split against unsplit on one device: parallel beam identical; cone beam max 6.3e-3 (Siddon) and 0.37 (interpolated)
  at the piece boundaries, the same on the CPU device.
- Page-locking: unsplit calls register nothing; split calls register the image once (urUSMImportExp, 28311552 bytes)
  and release it. No TIGRE:hostRegister warnings; the pageable fallback was not triggered.
- Repeated calls on the same array give identical results.
- pytest (test_sycl_low_memory, test_siddon_split, test_thin_shapes, test_projectors_gpu): 44 passed, 44 skipped
  (36 need `_Atb`, 8 need 2 GPUs).
