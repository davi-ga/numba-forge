"""Fake forge-optimised module: njit kernel + wrapper, exactly the shape the
forge pipeline emits."""

import numpy as np
from numba import njit


@njit
def _extract_qa_bits_jit(band_data, bit_location):
    return band_data & (1 << bit_location)


def extract_qa_bits(band_data, bit_location):
    band_data = np.asarray(band_data, dtype=np.uint16)
    bit_location = np.uint8(bit_location)
    out = _extract_qa_bits_jit(band_data, bit_location)
    return out.tolist() if hasattr(out, "tolist") else out


def create_index(index_name):
    """Simulated side-effectful function — must NEVER run at warmup."""
    raise AssertionError("side effect executed during boot!")
