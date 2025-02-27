import math
import pickle
from pathlib import Path
import warnings

import numpy as np
import torch

from .generate_primes import generate_message_primes, generate_scale_primes
from .security_parameters import maximum_qbits
from liberate.fhe.cache import cache
from . import errors 

# ------------------------------------------------------------------------------------------
# NTT parameter pre-calculation.
# ------------------------------------------------------------------------------------------
CACHE_FOLDER = cache.path_cache


def primitive_root_2N(q, N):
    _2N = 2 * N
    K = (q - 1) // _2N
    for x in range(2, N):
        g = pow(x, K, q)
        h = pow(g, N, q)
        if h != 1:
            break
    return g


def psi_power_series(psi, N, q):
    series = [1]
    for i in range(N - 1):
        series.append(series[-1] * psi % q)
    return series


def bit_rev_psi(q, logN):
    N = 2 ** logN
    psi = [primitive_root_2N(qi, N) for qi in q]
    # Bit-reverse index.
    ind = range(N)
    brind = [bit_reverse(i, logN) for i in ind]
    # The psi power and the indices are the same.
    return [pow(psi, brpower, q) for brpower in brind]


def psi_bank(q, logN):
    N = 2 ** logN
    psi = [primitive_root_2N(qi, N) for qi in q]
    ipsi = [pow(psii, -1, qi) for psii, qi in zip(psi, q)]
    psi_series = [psi_power_series(psii, N, qi) for psii, qi in zip(psi, q)]
    ipsi_series = [
        psi_power_series(ipsii, N, qi) for ipsii, qi in zip(ipsi, q)
    ]
    return psi_series, ipsi_series


def bit_reverse(a, nbits):
    format_string = f"0{nbits}b"
    binary_string = f"{a:{format_string}}"
    reverse_binary_string = binary_string[::-1]
    return int(reverse_binary_string, 2)


def bit_reverse_order_index(logN):
    N = 2 ** logN
    # Note that for a bit reversing, forward and backward permutations are the same.
    # i.e., don't worry about which direction.
    revi = np.array([bit_reverse(i, logN) for i in range(N)], dtype=np.int32)
    return revi


def get_psi(q, logN, my_dtype):
    np_dtype_dict = {
        np.int32: np.int32,
        np.int64: np.int64,
        30: np.int32,
        62: np.int64,
    }
    dtype = np_dtype_dict[my_dtype]
    psi, ipsi = psi_bank(q, logN)
    bit_reverse_index = bit_reverse_order_index(logN)
    psi = np.array(psi, dtype=dtype)[:, bit_reverse_index]
    ipsi = np.array(ipsi, dtype=dtype)[:, bit_reverse_index]
    return psi, ipsi


def paint_butterfly_forward(logN):
    N = 2 ** logN
    t = N
    painted_even = np.zeros((logN, N), dtype=np.bool_)
    painted_odd = np.zeros((logN, N), dtype=np.bool_)
    painted_psi = np.zeros((logN, N // 2), dtype=np.int32)
    for logm in range(logN):
        m = 2 ** logm
        t //= 2
        psi_ind = 0
        for i in range(m):
            j1 = 2 * i * t
            j2 = j1 + t - 1
            Sind = m + i
            for j in range(j1, j2 + 1):
                Uind = j
                Vind = j + t
                painted_even[logm, Uind] = True
                painted_odd[logm, Vind] = True
                painted_psi[logm, psi_ind] = Sind
                psi_ind += 1
    painted_eveni = np.where(painted_even)[1].reshape(logN, -1)
    painted_oddi = np.where(painted_odd)[1].reshape(logN, -1)
    return painted_eveni, painted_oddi, painted_psi


def paint_butterfly_backward(logN):
    N = 2 ** logN
    t = 1
    painted_even = np.zeros((logN, N), dtype=np.bool_)
    painted_odd = np.zeros((logN, N), dtype=np.bool_)
    painted_psi = np.zeros((logN, N // 2), dtype=np.int32)
    for logm in range(logN, 0, -1):
        level = logN - logm
        m = 2 ** logm
        j1 = 0
        h = m // 2
        psi_ind = 0
        for i in range(h):
            j2 = j1 + t - 1
            Sind = h + i
            for j in range(j1, j2 + 1):
                Uind = j
                Vind = j + t
                # Paint
                painted_even[level, Uind] = True
                painted_odd[level, Vind] = True
                painted_psi[level, psi_ind] = Sind
                psi_ind += 1
            j1 += 2 * t
        t *= 2
    painted_eveni = np.where(painted_even)[1].reshape(logN, -1)
    painted_oddi = np.where(painted_odd)[1].reshape(logN, -1)
    return painted_eveni, painted_oddi, painted_psi


# ------------------------------------------------------------------------------------------
# The context class.
# ------------------------------------------------------------------------------------------


class CkksContext:
    def __init__(
            self,
            buffer_bit_length=62,
            scale_bits=40,
            logN=15,
            num_scales=None,
            num_special_primes=2,
            sigma=3.2,
            uniform_ternary_secret=True,
            cache_folder=CACHE_FOLDER,
            security_bits=128,
            quantum="post_quantum",
            distribution="uniform",
            read_cache=True,
            save_cache=True,
            verbose=False,
            is_secured=True

    ):
        if not Path(cache_folder).exists():
            Path(cache_folder).mkdir(parents=True, exist_ok=True)

        self.generation_string = f"{buffer_bit_length}_{scale_bits}_{logN}_{num_scales}_" \
                                 f"{num_special_primes}_{security_bits}_{quantum}_" \
                                 f"{distribution}"

        self.is_secured = is_secured
        # Compose cache savefile name.
        savepath = Path(cache_folder) / Path(self.generation_string + ".pkl")

        if savepath.exists() and read_cache:
            with savepath.open("rb") as f:
                __dict__ = pickle.load(f)
                self.__dict__.update(__dict__)

            if verbose:
                print(
                    f"I have read in from the cached save file {savepath}!!!\n"
                )
                self.init_print()

            return

        # Transfer input parameters.
        self.buffer_bit_length = buffer_bit_length
        self.scale_bits = scale_bits
        self.logN = logN
        self.num_special_primes = num_special_primes
        self.cache_folder = cache_folder
        self.security_bits = security_bits
        self.quantum = quantum
        self.distribution = distribution
        # Sampling strategy.
        self.sigma = sigma
        self.uniform_ternary_secret = uniform_ternary_secret
        if self.uniform_ternary_secret:
            self.secret_key_sampling_method = "uniform ternary"
        else:
            self.secret_key_sampling_method = "sparse ternary"

        # dtypes.
        self.torch_dtype = {30: torch.int32, 62: torch.int64}[
            self.buffer_bit_length
        ]
        self.numpy_dtype = {30: np.int32, 62: np.int64}[self.buffer_bit_length]

        # Polynomial length.
        self.N = 2 ** self.logN

        # We set the message prime to of bit-length W-2.
        self.message_bits = self.buffer_bit_length - 2

        # Read in pre-calculated high-quality primes.
        try:
            message_special_primes = generate_message_primes(cache_folder=cache_folder)[self.message_bits][self.N]
        except KeyError as e:
            raise errors.NotFoundMessageSpecialPrimes(message_bit=self.message_bits, N=self.N)

        # For logN > 16, we need significantly more primes.
        how_many = 64 if self.logN < 16 else 128
        try:
            scale_primes = generate_scale_primes(cache_folder=cache_folder, how_many=how_many)[self.scale_bits, self.N]
        except KeyError as e:
            raise errors.NotFoundScalePrimes(scale_bits=self.scale_bits, N=self.N)

        # Compose the primes pack.
        # Rescaling drops off primes in --> direction.
        # Key switching drops off primes in <-- direction.
        # Hence, [scale primes, base message prime, special primes]
        self.max_qbits = int(
            maximum_qbits(self.N, security_bits, quantum, distribution)
        )
        base_special_primes = message_special_primes[: 1 + self.num_special_primes]

        # If num_scales is None, generate the maximal number of levels.
        try:
            if num_scales is None:
                base_special_bits = sum(
                    [math.log2(p) for p in base_special_primes]
                )
                available_bits = self.max_qbits - base_special_bits
                num_scales = 0
                available_bits -= math.log2(scale_primes[num_scales])
                while available_bits > 0:
                    num_scales += 1
                    available_bits -= math.log2(scale_primes[num_scales])

            self.num_scales = num_scales
            self.q = scale_primes[:num_scales] + base_special_primes
        except IndexError as e:
            raise errors.NotEnoughPrimes(scale_bits=self.scale_bits, N=self.N)

        # Check if security requirements are met.
        self.total_qbits = math.ceil(sum([math.log2(qi) for qi in self.q]))

        if self.total_qbits > self.max_qbits:
            if self.is_secured:
                raise errors.ViolatedAllowedQbits(
                    scale_bits=self.scale_bits, N=self.N, num_scales=self.num_scales,
                    max_qbits=self.max_qbits, total_qbits=self.total_qbits)
            else:
                warnings.warn(
                    f"Maximum allowed qbits are violated: "
                    f"max_qbits={self.max_qbits:4d} and the "
                    f"requested total is {self.total_qbits:4d}."
                )

        # Generate Montgomery parameters and NTT paints.
        self.generate_montgomery_parameters()
        self.generate_paints()

        if verbose:
            self.init_print()

        # Save cache.
        if save_cache:
            with savepath.open("wb") as f:
                pickle.dump(self.__dict__, f)

            if verbose:
                print(f"I have saved to the cached save file {savepath}!!!\n")

    def generate_montgomery_parameters(self):
        self.R = 2 ** self.buffer_bit_length
        self.R_square = [self.R ** 2 % qi for qi in self.q]
        self.half_buffer_bit_length = self.buffer_bit_length // 2
        self.lower_bits_mask = (1 << self.half_buffer_bit_length) - 1
        self.full_bits_mask = (1 << self.buffer_bit_length) - 1

        self.q_lower_bits = [qi & self.lower_bits_mask for qi in self.q]
        self.q_higher_bits = [
            qi >> self.half_buffer_bit_length for qi in self.q
        ]
        self.q_double = [qi << 1 for qi in self.q]

        self.R_inv = [pow(self.R, -1, qi) for qi in self.q]
        self.k = [
            (self.R * R_invi - 1) // qi
            for R_invi, qi in zip(self.R_inv, self.q)
        ]
        self.k_lower_bits = [ki & self.lower_bits_mask for ki in self.k]
        self.k_higher_bits = [
            ki >> self.half_buffer_bit_length for ki in self.k
        ]

    def generate_paints(self):
        self.N_inv = [pow(self.N, -1, qi) for qi in self.q]

        # psi and psi_inv.
        psi, psi_inv = get_psi(self.q, self.logN, self.buffer_bit_length)

        # Paints.
        (
            self.forward_even_indices,
            self.forward_odd_indices,
            forward_psi_paint,
        ) = paint_butterfly_forward(self.logN)
        (
            self.backward_even_indices,
            self.backward_odd_indices,
            backward_psi_paint,
        ) = paint_butterfly_backward(self.logN)

        # Pre-painted psi and ipsi.
        self.forward_psi = psi[..., forward_psi_paint.ravel()].reshape(
            -1, *forward_psi_paint.shape
        )
        self.backward_psi_inv = psi_inv[
            ..., backward_psi_paint.ravel()
        ].reshape(-1, *backward_psi_paint.shape)

    def init_print(self):
        print(f"""
I have received inputs:
        buffer_bit_length\t\t= {self.buffer_bit_length:,d}
        scale_bits\t\t\t= {self.scale_bits:,d}
        logN\t\t\t\t= {self.logN:,d}
        N\t\t\t\t= {self.N:,d}
        Number of special primes\t= {self.num_special_primes:,d}
        Number of scales\t\t= {self.num_scales:,d}
        Cache folder\t\t\t= '{self.cache_folder:s}'
        Security bits\t\t\t= {self.security_bits:,d}
        Quantum security model\t\t= {self.quantum:s}
        Security sampling distribution\t= {self.distribution:s}
        Number of message bits\t\t= {self.message_bits:,d}
        In total I will be using '{self.total_qbits:,d}' bits out of available maximum '{self.max_qbits:,d}' bits.
        And is it secured?\t\t= {self.is_secured}
My RNS primes are {self.q}."""
              )
