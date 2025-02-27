# -*- coding: utf-8 -*-
#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import typing
from loguru import logger
import inspect
from functools import wraps


def singleton(class_):
    class class_w(class_):
        _instance = None

        def __new__(cls, *args, **kwargs):
            if class_w._instance is None:
                class_w._instance = super(class_w, cls).__new__(
                    cls, *args, **kwargs
                )


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


def patch(func=None, *, name=None, overwrite=True):
    """Patch a function into a class type

    Args:
        func (Function): A function that takes at least one argument with a specific class type 'self:YourClass'
        name (str, optional): The name to assign to the method in the class. Defaults to the function's name.
        overwrite (bool, optional): Whether to overwrite an existing method in the class. Defaults to True.

    Returns:
        function: The patched function
    """
    if func is None:
        return lambda f: patch(f, name=name, overwrite=overwrite)

    # Extract the class from the first parameter's type annotation
    cls = next(iter(func.__annotations__.values()))
    method_name = name or func.__name__

    # Check if the method already exists in the class
    if hasattr(cls, method_name):
        if not overwrite:
            # Do not overwrite; return the original function unmodified
            return func
        else:
            # Overwrite; remove the existing method from the class
            logger.warning(
                f"Overwriting existing method {method_name} in class {cls.__name__}"
            )

    # Update function metadata
    func.__qualname__ = f"{cls.__name__}.{method_name}"
    func.__module__ = cls.__module__

    # Patch the function into the class
    setattr(cls, method_name, func)
    return func


def initonly(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current call stack
        stack = inspect.stack()
        # Ensure there's a caller frame; typically, __init__ should be the immediate caller.
        # You might need to adjust the depth if there are intermediary wrappers.
        if len(stack) < 2 or stack[1].function != "__init__":
            raise RuntimeError(
                f"Method '{func.__name__}' can only be called from __init__"
            )
        return func(*args, **kwargs)

    return wrapper

STRICT_TYPE_CHECKING = False

def strictype(func):
    """
    A decorator that checks the types of the arguments passed to a function.
    
    Warning, this will cause cpu bound in some cases, use with caution.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not STRICT_TYPE_CHECKING:
            return func(*args, **kwargs)
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