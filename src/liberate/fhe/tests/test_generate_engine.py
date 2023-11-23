import pytest

from liberate import fhe


@pytest.fixture()
def ckks_engine(
    devices: list[str] = None,
    logN: int = 15,
    scale_bits: int = 40,
    read_cache: bool = True,
):
    """
        generate ckks_engine
    @param devices:
    @param logN:
    @param scale_bits:
    @param read_cache:
    @return:
    """
    ctx_params = {
        "logN": logN,
        "scale_bits": scale_bits,
        "security_bits": 128,
        "num_scales": None,
        "num_special_primes": 2,
        "buffer_bit_length": 62,
        "sigma": 3.2,
        "uniform_tenary_secret": True,
        "cache_folder": "cache/",
        "quantum": "post_quantum",
        "distribution": "uniform",
        "read_cache": read_cache,
        "save_cache": True,
    }
    engine = fhe.ckks_engine(devices=devices, verbose=False, **ctx_params)
    return engine


scales = [x for x in range(20, 50, 5)]
logNs = [x for x in range(14, 17)]
test_cases = [
    (logN, scale, True)
    for logN in [x for x in range(14, 17)]  # logNs
    for scale in [x for x in range(20, 50, 5)]  # scales
]


@pytest.mark.parametrize("ckks_engine", test_cases, indirect=["ckks_engine"])
def test_make_engine(ckks_engine):
    """
        test for generate ckks engine
    @param ckks_engine:
    @return:
    """
    assert isinstance(ckks_engine, fhe.ckks_engine)
