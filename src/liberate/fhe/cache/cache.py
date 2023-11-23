import glob
import os
from ..context import generate_primes

path_cache = os.path.abspath(__file__).replace("cache.py", "resources")


# logN_N_M = os.path.join(path_cache, "logN_N_M.pkl")
# message_special_primes = os.path.join(path_cache, "message_special_primes.pkl")
# scale_primes = os.path.join(path_cache, "scale_primes.pkl")


def clean_cache(path=None):
    if path is None:
        path = path_cache
    files = glob.glob(os.path.join(path, "*.pkl"))
    for file in files:
        try:
            os.unlink(file)
        except Exception as e:
            print(e)
            pass
    return


def generate_cache(path=None):
    if path is None:
        path = path_cache
    # Read in pre-calculated high-quality primes.
    _ = generate_primes.generate_message_primes(cache_folder=path)
    _ = generate_primes.generate_scale_primes(cache_folder=path)
    return
