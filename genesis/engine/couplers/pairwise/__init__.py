"""
The couplings between pairs of solvers, one System per pair of coupled solvers.

Each one exists when both solvers hold entities and the coupler options (LegacyCouplerOptions) enable it. The couplings
of a substep run in the order they are registered here.
"""

from genesis.engine.pipeline import register_system

from .fem_mpm import FEMMPM
from .fem_sph import FEMSPH
from .fem_surface import FEMSurface
from .mpm_particles import MPMPBD, MPMSPH
from .rigid_fem import RigidFEM, RigidFEMAttachment
from .rigid_mpm import RigidMPM, RigidMPMAttachment, RigidMPMCPIC, RigidMPMHybrid
from .rigid_pbd import RigidPBD
from .rigid_sph import RigidSPH

for system_cls in (
    RigidMPM,
    RigidMPMCPIC,
    RigidMPMHybrid,
    RigidMPMAttachment,
    MPMSPH,
    MPMPBD,
    RigidSPH,
    RigidPBD,
    FEMSurface,
    RigidFEMAttachment,
    RigidFEM,
    FEMMPM,
    FEMSPH,
):
    register_system(system_cls)
