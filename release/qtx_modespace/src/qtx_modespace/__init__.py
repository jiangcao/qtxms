"""Mode-space basis for QuaTrEx devices: level-1 (local) and direct level-2 (global) density-matrix modes per slab or
tile, spurious lead-band cleanup, mode-space transmission and FGMRES with a CAP-Schwarz smoother."""
__version__ = "0.2.0"

from .quatrex import read_quatrex
from .core import Device, Coarse, CSolver, injection, rho_contour, top_modes
from .clean import LeadCell, RMS, clean_slab
from .level1 import partition, build_level1, local_rho, embedding, Geometry
from .schwarz import CapSchwarz, fgmres_block, two_level
