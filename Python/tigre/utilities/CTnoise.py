import numpy as np

# SYCL-only mode: Check if RNG extension is available
try:
    import _RandomNumberGenerator as RNG
    HAS_RNG_EXTENSION = True
except ImportError:
    # SYCL-only mode - use numpy random number generator instead
    RNG = None
    HAS_RNG_EXTENSION = False
    print("⚠️  CUDA RandomNumberGenerator not available - using numpy RNG")

def add(projections, Gaussian=None, Poisson=None):

    if Poisson is not None:
        if not np.isscalar(Poisson):
            raise ValueError(
                "Poisson value should be an scalar, is " + str(type(Poisson)) + " instead."
            )
    else:
        Poisson = np.ceil(np.log2(np.max(np.abs(projections))))  # nextpow2
    if Gaussian is not None:
        if not isinstance(Gaussian, np.ndarray):
            raise ValueError(
                "Gaussian value should be an array, is " + str(type(Gaussian)) + " instead."
            )
        if Gaussian.shape != (2,):
            raise ValueError("Gaussian shape should be 1x2, is " + str(Gaussian.shape) + "instead.")
    else:
        Gaussian = np.array([0, 0.5])
    max_proj = np.max(projections)
    projections = Poisson * np.exp(-projections / max_proj)

    # Use RNG extension if available, otherwise fallback to numpy
    if HAS_RNG_EXTENSION:
        projections = RNG.add_noise(projections, Gaussian[0], Gaussian[1])
    else:
        # Fallback implementation using numpy
        # Apply Poisson noise
        projections = np.random.poisson(projections)
        # Apply Gaussian noise: mean=Gaussian[0], std=Gaussian[1]
        if Gaussian[1] > 0:  # Only add noise if std > 0
            projections = projections + np.random.normal(Gaussian[0], Gaussian[1], projections.shape)

    projections = -np.log(projections / Poisson) * max_proj
    projections = np.float32(projections)
    return projections
