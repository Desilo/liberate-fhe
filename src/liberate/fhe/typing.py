import typing
from typing import Dict, List, Union
import inspect
from functools import wraps
import numpy as np
from loguru import logger
from torch import Tensor


class DataStruct:

    # data: Union[list, tuple]
    # include_special: bool
    # ntt_state: bool
    # montgomery_state: bool
    # origin: str
    # level: int
    # hash: str

    def __init__(
        self,
        data: Union[list, tuple],
        include_special: bool,
        ntt_state: bool,
        montgomery_state: bool,
        level: int,
        hash: str,
        description: str = None,
        *args,
        **kwargs,
    ):
        """
        Data structure:
        - data: the data in tensor format
        - include_special: Boolean, including the special prime channels or not.
        - ntt_state: Boolean, whether if the data is ntt transformed or not.
        - montgomery_state: Boolean, whether if the data is in the Montgomery form or not.
        - origin: String, where did this data came from - cipher text, secret key, etc.
        - level: Integer, the current level where this data is situated.
        - hash: String, a SHA256 hash of the input settings and the prime numbers used to RNS decompose numbers.
        - version: String, version number.
        """
        self.data = data
        self.include_special = include_special
        self.ntt_state = ntt_state
        self.montgomery_state = montgomery_state
        self.level = level
        self.hash = hash
        self.description = description

    @classmethod
    def clone_tensor_recursive(cls, data):
        """Recursively clone tensors in the data structure.
        Args:
            data: The data structure to clone.
        Returns:
            The cloned data structure.
        """
        # Recursively clone tensors in the data structure
        if isinstance(data, Tensor):
            return data.clone()
        elif isinstance(data, np.ndarray):  # plaintext src
            return data.copy()
        elif isinstance(data, list):
            return [cls.clone_tensor_recursive(item) for item in data]
        elif isinstance(data, tuple):  # legacy datastruct uses tuple
            return tuple(cls.clone_tensor_recursive(item) for item in data)
        elif isinstance(data, dict):  # plaintext cache
            return {
                key: cls.clone_tensor_recursive(value)
                for key, value in data.items()
            }
        elif isinstance(data, DataStruct):
            return data.clone()
        else:
            # raise TypeError(f"Unsupported data type to clone: {type(data)}")
            logger.warning(
                f"Unsupported data type to clone: {type(data)}, returning the original data."
            )
            return data

    def clone(self):
        """Clone the data structure.

        Returns:
            DataStruct or its subclasses: A new instance of the same class with cloned data.
        """
        cls = self.__class__  # Get the class of the current instance
        return cls(
            data=cls.clone_tensor_recursive(self.data),
            include_special=self.include_special,
            ntt_state=self.ntt_state,
            montgomery_state=self.montgomery_state,
            level=self.level,
            hash=self.hash,
            description=self.description,
        )

    @classmethod
    def wrap(cls, another: "DataStruct"):
        """Wrap another data structure into a new instance of the same class.
        Args:
            another (DataStruct): The data structure to wrap.
        Returns:
            DataStruct or its subclasses: A new instance of the same class with the same attributes as `another`.
        """
        return cls(
            data=another.data,
            include_special=another.include_special,
            ntt_state=another.ntt_state,
            montgomery_state=another.montgomery_state,
            level=another.level,
            hash=another.hash,
            description=another.description,
        )

    @classmethod
    def get_device_of_tensor(cls, data):
        """Get the device of the tensor in the data structure.
        Args:
            data: The data structure to check.
        Returns:
            The device of the tensor.
        """
        # Recursively get the device of the tensor in the data structure
        if isinstance(data, Tensor):
            return data.device
        elif isinstance(data, list):
            return cls.get_device_of_tensor(data[0])
        elif isinstance(data, tuple):
            return cls.get_device_of_tensor(data[0])
        elif isinstance(data, dict):
            return cls.get_device_of_tensor(list(data.values())[0])
        elif isinstance(data, DataStruct):
            return cls.get_device_of_tensor(data.data)
        else:
            logger.warning(
                f"Unsupported data type to get device: {type(data)}, will return 'cpu'."
            )
            return "cpu"

    @property
    def device(self):
        """Get the device of the data structure.
        Returns:
            The device of the data structure.
        """
        return self.get_device_of_tensor(self.data)

    @classmethod
    def move_tensor_to_device_recursive(
        cls, data, device: str, non_blocking=True
    ):
        """Recursively move tensors in the data structure to a specified device.
        Args:
            data: The data structure to move.
            device: The target device.
        Returns:
            The data structure moved to the specified device.
        """
        # Recursively move tensors in the data structure to the specified device
        if isinstance(data, Tensor):
            return data.to(device, non_blocking=non_blocking)
        elif isinstance(data, list):
            return [
                cls.move_tensor_to_device_recursive(item, device)
                for item in data
            ]
        elif isinstance(data, tuple):  # legacy datastruct uses tuple
            return tuple(
                cls.move_tensor_to_device_recursive(item, device)
                for item in data
            )
        elif isinstance(data, dict):  # plaintext cache
            return {
                key: cls.move_tensor_to_device_recursive(value, device)
                for key, value in data.items()
            }
        elif isinstance(data, DataStruct):
            return data.to_device(device)
        else:
            logger.warning(
                f"Unsupported data type to move to device: {type(data)}, returning the original data."
            )
            return data

    def to_device(self, device: str, non_blocking=True):
        """Move the data structure to a specified device.
        Args:
            device: The target device.
        Returns:
            DataStruct or its subclasses: A new instance of the same class with data moved to the specified device.
        """
        cls = self.__class__
        return cls(
            data=cls.move_tensor_to_device_recursive(
                data=self.data, device=device, non_blocking=non_blocking
            ),
            include_special=self.include_special,
            ntt_state=self.ntt_state,
            montgomery_state=self.montgomery_state,
            level=self.level,
            hash=self.hash,
            description=self.description,
        )

    def to(self, device: str, non_blocking=True):
        # alias for to_device
        return self.to_device(device, non_blocking)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, include_special={self.include_special}, ntt_state={self.ntt_state}, montgomery_state={self.montgomery_state}, level={self.level}, description={self.description}, hash={self.hash})"

    def __str__(self):
        return self.__repr__()  # todo for better readability


# ================== #
#  Cipher Structures #
# ================== #


class Ciphertext(DataStruct):
    pass


class CiphertextTriplet(DataStruct):
    # todo does CiphertextTriplet even exits ntt_state and montgomery_state?
    pass


# ================== #
#  Key Structures    #
# ================== #


class SecretKey(DataStruct):
    # todo does secret key even exits ntt_state and montgomery_state?
    pass


class EvaluationKey(SecretKey):
    pass


class PublicKey(DataStruct):
    pass


class KeySwitchKey(DataStruct):
    pass


class RotationKey(KeySwitchKey):
    delta: int = None
    pass


class GaloisKey(DataStruct):
    pass


class ConjugationKey(DataStruct):
    pass


# ================== #
#  Plaintext Cache   #
# ================== #


class Plaintext(DataStruct):
    def __init__(
        self,
        src: Union[list, tuple],
        cache: Dict[int, list] = {},
        *args,
        **kwargs,
    ):
        self.src = src
        self.cache = cache

    def clone(self):
        return Plaintext(
            src=self.__class__.clone_tensor_recursive(self.src),
            cache=self.__class__.clone_tensor_recursive(self.cache),
        )

    @property
    def data(self):
        raise NotImplementedError("data is not implemented for Plaintext")

    @property
    def include_special(self):
        raise NotImplementedError(
            "include_special is not implemented for Plaintext"
        )

    @property
    def ntt_state(self):
        raise NotImplementedError("ntt_state is not implemented for Plaintext")

    @property
    def montgomery_state(self):
        raise NotImplementedError(
            "montgomery_state is not implemented for Plaintext"
        )

    @property
    def level(self):
        raise NotImplementedError("level is not implemented for Plaintext")

    @property
    def hash(self):
        raise NotImplementedError("hash is not implemented for Plaintext")

# ================== #
#  Helper Functions  #
# ================== #


def strictype(func):
    """
    A decorator that checks the types of the arguments passed to a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints for the function
        hints = typing.get_type_hints(func)
        # Bind the passed arguments to the function signature
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        # Iterate over all arguments and check their types
        for name, value in bound.arguments.items():
            if name in hints:
                expected_type = hints[name]
                # print(expected_type)
                try:
                    _ = isinstance(value, expected_type)
                except Exception:
                    # logger.warning("error in strictype")
                    pass  # todo TypeError: Subscripted generics cannot be used with class and instance checks
                else:
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{name}' must be of type {expected_type}, "
                            f"but got {type(value)}."
                        )
        return func(*args, **kwargs)

    return wrapper
