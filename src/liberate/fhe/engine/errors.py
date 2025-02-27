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
        self.message_error = (
            f"""The input secret key must include special prime channels."""
        )
        super().__init__(self.message_error)

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error

class DifferentTypeError(Exception):
    def __init__(self, a, b):
        self.message_error = (
            f"The type of two parameter should be the same, but they are '{a}' and '{b}'."
        )

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error

class NTTStateError(Exception):
    def __init__(self, expected):
        self.message_error = f"""The data_struct ntt_state should be a '{expected}', but it is '{not expected}'."""

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error
    
class MontgomeryStateError(Exception):
    def __init__(self, expected):
        self.message_error = f"""The data_struct montgomery_state should be a '{expected}', but it is '{not expected}'."""

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error

class MaximumLevelError(Exception):
    def __init__(self, level, level_max):
        self.message_error = f"""The number of multiplications available for this cipher text is fully depleted. I cannot proceed further. maximum : {level_max:2d}, now : {level:2d}""".strip()

    def __repr__(self):
        return repr(self.message_error)

    def __str__(self):
        return self.message_error