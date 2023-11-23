from typing import NamedTuple
from .version import VERSION


class data_struct(NamedTuple):
    """
        - Data structure.
        - data: the data in tensor format
        - include_special: Boolean, including the special prime channels or not.
        - ntt_state: Boolean, whether if the data is ntt transformed or not.
        - montgomery_state: Boolean, whether if the data is in the Montgomery form or not.
        - origin: String, where did this data came from - cipher text, secret key, etc.
        - level: Integer, the current level where this data is situated.
        - hash: String, a SHA256 hash of the input settings and the prime numbers used to RNS decompose numbers.
        - version: String, version number.
    """
    data: tuple | list
    include_special: bool
    ntt_state: bool
    montgomery_state: bool
    origin: str
    level: int
    hash: str
    version: str = VERSION
