class NotFoundMessageSpecialPrimes(Exception):
    def __init__(self, message_bit, N):
        self.message_error = f"""Can't find message_bit = {message_bit:3<d} and N = {N:6<d}""".strip()
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
    
    
class NotEnoughPrimes(Exception):
    def __init__(self, scale_bits, N):
        self.message_error = f"""Not enough scale bit at scale bits = {scale_bits:3<d} and N = {N:6<d}""".strip()
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
    
class NotFoundScalePrimes(Exception):
    def __init__(self, scale_bits, N):
        self.message_error = f"""Can't find scale bits = {scale_bits:3<d} and N = {N:6<d}""".strip()
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error
