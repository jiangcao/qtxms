"""Mode-space basis for QuaTrEx devices: direct level-2 slab density-matrix modes with spurious lead-band cleanup."""
__version__ = "0.2.0.dev0"

from .quatrex import read_quatrex
from .core import Device, Coarse, CSolver, injection, rho_contour, top_modes
from .clean import LeadCell, RMS, clean_slab
