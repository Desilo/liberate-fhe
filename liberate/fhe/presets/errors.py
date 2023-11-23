from functools import wraps
import logging


def log_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"[Error] Error in {func.__name__} : {e}")
            raise

    return wrapper


class TestException(Exception):
    def __init__(self):
        message_error = "test error"
        super().__init__(message_error)


class NotFoundMessageSpecialPrimes(Exception):
    def __init__(self, message_bit, N):
        self.message_error = f"""Can't find message_bit = {message_bit:3<d} and N = {N:6<d}""".strip()
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class NotFoundScalePrimes(Exception):
    def __init__(self, scale_bits, N):
        self.message_error = f"""Can't find scale bits = {scale_bits:3<d} and N = {N:6<d}""".strip()
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class NotEnoughPrimes(Exception):
    def __init__(self, scale_bits, N):
        self.message_error = f"""Not enough scale bit at scale bits = {scale_bits:3<d} and N = {N:6<d}""".strip()
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class ViolatedAllowedQbits(Exception):
    def __init__(self, scale_bits, N, num_scales, max_qbits, total_qbits):
        self.message_error = f"""Maximum allowed qbits are violated:
max_qbits={max_qbits:4d} and the
requested total is {total_qbits:4d}.
scale_bits = {scale_bits:6<d}, N = {N:3<d} and num_scales = {num_scales:4<d}
"""
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class NotEnoughPrimesForBiasGuard(Exception):
    def __init__(self, bias_guard, num_special_primes):
        self.message_error = f"""Guarding against biased overflow\nrequires the number of special prime\nchannels greater than 2.\nbias_guard = {bias_guard}, num_special_primes = {num_special_primes}"""
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class NotFindBufferBitLength(Exception):
    def __init__(self, buffer_bit_length):
        self.message_error = f"""Can't find buffer length bit {buffer_bit_length}.\nYou can only choose between 30 or 62."""
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class SecretKeyNotIncludeSpecialPrime(Exception):
    def __init__(self):
        self.message_error = f"""The input secret key must include special prime channels."""
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class DifferentTypeError(Exception):
    def __init__(self, a, b):
        self.message_error = f"""The data type are different. {a}, {b}"""

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class NotMatchType(Exception):
    def __init__(self, origin, to):
        self.message_error = f"""The data_struct origin should be a '{to}', but it is '{origin}'."""

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class NotMatchDataStructState(Exception):
    def __init__(self, origin: str):
        self.message_error = f"""Wrong format of the source {origin} detected, \nApply ntt and the montgomery transformation to the data."""

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class MaximumLevelError(Exception):
    def __init__(self, level, level_max):
        self.message_error = f"""The number of multiplications available
for this cipher text is fully depleted. 
I cannot proceed further.
maximum : {level_max:2d}, now : {level:2d}""".strip()

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error


class DeviceSelectError(Exception):
    def __init__(self):
        self.message_error = "To download data to the CPU, it must already be in a GPU!!!"

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error
