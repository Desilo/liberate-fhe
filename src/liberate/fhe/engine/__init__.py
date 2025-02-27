from .ckks_engine import CkksEngine
from . import extension

presets = {
    "bronze": {
        "logN": 14,
        "num_special_primes": 1,
        "devices": [0],
        "scale_bits": 40,
        "num_scales": None,
    },
    "silver": {
        "logN": 15,
        "num_special_primes": 2,
        "devices": [0],
        "scale_bits": 40,
        "num_scales": None,
    },
    "gold": {
        "logN": 16,
        "num_special_primes": 4,
        "devices": None,
        "scale_bits": 40,
        "num_scales": None,
    },
    "platinum": {
        "logN": 17,
        "num_special_primes": 6,
        "devices": None,
        "scale_bits": 40,
        "num_scales": None,
    },
}

__all__ = ["presets", "CkksEngine", "extension"]