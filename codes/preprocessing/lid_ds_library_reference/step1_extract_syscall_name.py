"""
REFERENCE COPY — not written by us.

Original file: external/LID-DS/algorithms/features/impl/syscall_name.py
(part of the upstream LID-DS framework: https://github.com/LID-DS/LID-DS)

Included here, unmodified, purely for transparency/documentation, since
external/LID-DS is excluded from this repo via .gitignore (it's a 220MB+
nested git clone — see README's Environment section). This is the exact
code our preprocessing.py pipeline imports and depends on.

WHAT THIS DOES (step 1 of 3):
Given one raw syscall event, pull out just its NAME (e.g. "read", "open",
"fcntl") and discard everything else about it (timestamp, PID, arguments,
return value, etc.). This is the very first step of the name-only feature
pipeline — the syscall name is the only thing that survives into later
steps (see step2 and step3 in this same folder).
"""
import typing

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class SyscallName(BuildingBlock):

    def __init__(self):
        super().__init__()

    def _calculate(self, syscall: Syscall):
        """
        calculate name of syscall
        """
        return syscall.name()

    def depends_on(self):
        return []
