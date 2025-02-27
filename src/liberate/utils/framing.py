# -*- coding: utf-8 -*-
#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust
# Date:   20230319

import inspect
import types
from dataclasses import dataclass
from os import path
from typing import Union


def get_frame_traceback(stack_offset=1):
    stack = inspect.stack()
    stack_offset = len(stack) - 1 if stack_offset >= len(stack) else stack_offset
    return stack[stack_offset]


def get_frame_func_name_traceback(stack_offset=1):
    frame = get_frame_traceback(stack_offset + 1)
    func = frame.function
    return None if func == "<module>" else func


def get_frame_class_traceback(stack_offset=1):
    frame = get_frame_traceback(stack_offset + 1)
    _locals = frame[0].f_locals
    if "self" in _locals:
        return _locals["self"].__class__
    return None


def get_frame_module_traceback(stack_offset=1) -> Union[types.ModuleType, None]:
    frame = get_frame_traceback(stack_offset + 1)
    module = inspect.getmodule(frame[0])
    return module


def get_frame_filepath_traceback(stack_offset=1):
    frame = get_frame_traceback(stack_offset + 1)
    return frame.filename


@dataclass
class TracebackInfo:
    frame = None
    lineno: int = None
    func_name: str = None
    locals = None
    class_obj = None
    class_name: str = None
    module = None
    module_name: str = None
    filepath: str = None
    filename: str = None

    @classmethod
    def parse(cls, frame) -> "TracebackInfo":
        stacktrace = TracebackInfo()
        stacktrace.frame = frame
        stacktrace.lineno = frame.lineno
        stacktrace.func_name = frame.function if frame.function != "<module>" else None
        stacktrace.locals = frame[0].f_locals
        if "self" in stacktrace.locals:
            stacktrace.class_obj = stacktrace.locals["self"].__class__
            if stacktrace.class_obj:
                stacktrace.class_name = stacktrace.class_obj.__name__
        module: Union[types.ModuleType, None] = inspect.getmodule(frame[0])
        stacktrace.module = module
        stacktrace.module_name = module.__name__ if module else None
        stacktrace.filepath = path.abspath(frame.filename)
        stacktrace.filename = path.basename(frame.filename)
        return stacktrace

    @property
    def last_describable(self):
        for identity in [self.func_name, self.class_name, self.module_name, self.filepath, None]:
            if identity is not None:
                return identity

    @property
    def strid(self):
        return "-".join(self.as_str_sequence(lineno=True))

    @property
    def json(self):
        return {
            "file": f"{self.filepath}",
            "modlue": f"{self.module_name}",
            "class": f"{self.class_name}",
            "function": f"{self.func_name}",
        }

    def format(self, fmt: str):
        result = (
            fmt.replace(r"%F", self.filepath or "_")
            .replace(r"%m", self.module_name or "_")
            .replace(r"%c", self.class_name or "_")
            .replace(r"%f", self.func_name or "_")
            .replace(r"%l", str(self.lineno))
        )
        return result

    def as_str_sequence(self, lineno=False):
        return [
            prop
            for prop in [
                self.filepath,
                str(self.lineno) if lineno else None,
                self.module_name,
                self.class_name,
                self.func_name,
            ]
            if prop is not None
        ]

    def __eq__(self, __value: object) -> bool:
        assert isinstance(__value, TracebackInfo), f"cannot compare {__value} as a {TracebackInfo}"
        return (
            self.filepath == __value.filepath
            and self.module_name == __value.module_name
            and self.class_name == __value.class_name
            and self.func_name == __value.func_name
        )

    def __repr__(self) -> str:
        return "\n".join([f"{k}:\t{v}" for k, v in self.json.items()])


def get_caller_info_traceback(stack_offset=1):
    frame = get_frame_traceback(stack_offset + 1)
    return TracebackInfo.parse(frame)
