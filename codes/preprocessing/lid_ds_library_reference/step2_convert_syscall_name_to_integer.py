"""
REFERENCE COPY — not written by us.

Original file: external/LID-DS/algorithms/features/impl/int_embedding.py
(part of the upstream LID-DS framework: https://github.com/LID-DS/LID-DS)

Included here, unmodified, purely for transparency/documentation, since
external/LID-DS is excluded from this repo via .gitignore (it's a 220MB+
nested git clone — see README's Environment section). This is the exact
code our preprocessing.py pipeline imports and depends on.

WHAT THIS DOES (step 2 of 3):
Takes the syscall NAME string from step1 (e.g. "read") and assigns it a
whole-number ID. The first time a name is ever seen during training, it
gets the next free integer (starting at 1 — 0 is reserved for "unknown",
used for any syscall name that only shows up later, in test/validation
data, and was never seen during training). This dictionary — name -> ID —
is exactly what's saved in our outputs/phase3_preprocessing/*_vocabulary.csv
files, and it's what turns a human-readable trace into the plain integers
you see in preprocessed data/*_ngrams.csv.zip.
"""
from dataloader.syscall import Syscall

from algorithms.building_block import BuildingBlock
from algorithms.features.impl.syscall_name import SyscallName


class IntEmbedding(BuildingBlock):
    """
        convert system call name to unique integer

        Params:
        building_block: BB which should be embedded as int
    """

    def __init__(self, building_block: BuildingBlock = None):
        super().__init__()
        self._syscall_dict = {}
        if building_block is None:
            building_block = SyscallName()
        self._dependency_list = [building_block]

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """
            takes one syscall and assigns integer
            integer is current length of syscall_dict
            keep 0 free for unknown syscalls
        """
        bb_value = self._dependency_list[0].get_result(syscall)
        if bb_value not in self._syscall_dict:
            self._syscall_dict[bb_value] = len(self._syscall_dict) + 1

    def _calculate(self, syscall: Syscall):
        """
            transforms given building_block to integer
        """
        bb_value = self._dependency_list[0].get_result(syscall)
        return self._syscall_dict.get(bb_value, 0)
