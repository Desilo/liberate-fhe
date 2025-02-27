# -*- coding: utf-8 -*-
#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import math
import torch
from typing import Any, Callable, Union
from liberate.fhe.typing import *
from loguru import logger

from .framing import get_caller_info_traceback


def copy_some_datastruct(src):
    if isinstance(src, DataStruct):
        return src.clone()
    if isinstance(src, torch.Tensor):
        return src.clone()
    if isinstance(src, np.ndarray):
        return src.copy()
    if isinstance(src, (list, tuple)):
        return [copy_some_datastruct(d) for d in src]
    if isinstance(src, dict):
        return {k: copy_some_datastruct(v) for k, v in src.items()}
    else:
        logger.warning(
            f"Unknown type: {type(src)} on copy. Will return the original."
        )
        return src


def calculate_tensor_size_in_bytes(tensor: torch.Tensor):
    shape = tensor.shape
    element_size = tensor.element_size()
    total_size = element_size
    for dim in shape:
        total_size *= dim
    return total_size


def calculate_ckks_cipher_datastruct_size_in_list_recursive(
    list_or_cipher: Union[tuple, list, torch.Tensor, DataStruct]
):
    if isinstance(list_or_cipher, DataStruct):
        return calculate_ckks_cipher_datastruct_size_in_list_recursive(
            list_or_cipher.data
        )
    elif isinstance(list_or_cipher, (list, tuple)):
        total_size = sum(
            [
                calculate_ckks_cipher_datastruct_size_in_list_recursive(d)
                for d in list_or_cipher
            ]
        )
    elif isinstance(list_or_cipher, torch.Tensor):
        return calculate_tensor_size_in_bytes(list_or_cipher)
    else:
        raise ValueError(f"Unknown type: {type(list_or_cipher)}")

    return total_size


class VariesByIndex(dict):
    debug = False

    def __getitem__(self, index: Any):
        if index not in self:
            if self.debug:
                caller_info = get_caller_info_traceback(stack_offset=2)
                logger.warning(
                    f"in {caller_info.last_describable} first time access to index {index}"
                )
            # set the value
            if self.origin is not None:
                self[index] = self.transform_fn(self.origin, index)
            else:
                self[index] = self.transform_fn(index)
        return super().__getitem__(index)

    def __init__(self, origin: Any, transform_fn: Callable, name: str):
        self.origin = origin
        self.transform_fn = transform_fn
        self.name = name


def next_power_of_n(x: int, n: int):
    return n ** math.ceil(math.log(x, n))


def next_power_of_2(n: int):
    """
    Return the smallest power of 2 greater than or equal to n
    Copied from triton code
    """
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def next_multiple_of_n(x: int, n: int):
    return n * math.ceil(x / n)


def decompose_with_power_of_n(a: int, n: int, return_expo=True) -> list[int]:
    result = []
    while a > 1:
        b = math.floor(math.log(a, n))
        c = int(n**b)
        if not return_expo:
            b = c
        result.append(b)
        a = a - c
    if a >= 1:
        result.append(a)
    result.reverse()
    return result


def decompose_with_power_of_2(a: int, return_expo=True) -> list[int]:
    return decompose_with_power_of_n(a, 2, return_expo)
