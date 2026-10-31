"""
The couplings between pairs of solvers, one System per pair of coupled solvers.

Each one exists when both solvers hold entities and the coupler options (LegacyCouplerOptions) enable it. The order of the
imports is the order the couplings of a substep run in.
"""

from .rigid_mpm import RigidMPM
from .mpm_sph import MPMSPH
from .mpm_pbd import MPMPBD
from .rigid_sph import RigidSPH
from .rigid_pbd import RigidPBD
from .fem_surface import FEMSurface
from .rigid_fem import RigidFEM, RigidFEMAttachment
from .fem_mpm import FEMMPM
from .fem_sph import FEMSPH
