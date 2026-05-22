"""
pyrespect_freq
--------------
Extract continuous and discrete relaxation spectra from frequency-domain
G*(w) data.

Public API
----------
    from pyrespect_freq import ReSpect, ReSpectConfig
"""

from .solver import ReSpect
from .config import ReSpectConfig

__all__ = ["ReSpect", "ReSpectConfig"]
__version__ = "2.0.0"
