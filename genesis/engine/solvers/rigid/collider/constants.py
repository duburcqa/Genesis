"""
Constants and enums for the collider module.
"""

from enum import IntEnum


class RETURN_CODE(IntEnum):
    """
    Return codes for the general subroutines used in GJK and EPA algorithms.
    """

    SUCCESS = 0
    FAIL = 1


class GJK_RETURN_CODE(IntEnum):
    """
    Return codes for the GJK algorithm.
    """

    SEPARATED = 0
    INTERSECT = 1
    NUM_ERROR = 2


class PORTAL_STATUS(IntEnum):
    """
    What the penetration depth of a contact is worth, and whether the portal behind it may be reused (perturbation
    reconstruction, EPA seeding). Each value names the depth rather than the portal's health, since that is what every
    consumer decides on.

    NONE: no portal exists. The contact came from a routine that computes it in closed form (plane, capsule, sphere) or
    from the MPR centres fallback, so the depth stands on its own and there is nothing for a refinement to improve. This
    is the value a slot carries when no detection wrote it, so an unwritten slot reads as "no portal" and never as a
    verdict about one.
    UNCONVERGED: MPR reached its iteration cap without a portal converging, so the depth means nothing.
    EXTRAPOLATED: the portal converged, but the origin's projection falls so far beyond the triangle that the depth is
    read off an extrapolation of its plane, or the triangle is degenerate. Untrustworthy.
    LOWER_BOUND: the origin's projection falls just outside the triangle, so the depth is a valid lower bound of the
    true one (Theorem 4.3). Trustworthy as a bound, but the portal is not the exact contact face.
    EXACT: the portal converged and the origin projects inside it, so the depth is exact (Theorem 4.2). The only status
    whose portal may be reused.
    """

    NONE = 0
    UNCONVERGED = 1
    EXTRAPOLATED = 2
    LOWER_BOUND = 3
    EXACT = 4


class EPA_POLY_INIT_RETURN_CODE(IntEnum):
    """
    Return codes for the EPA polytope initialization.
    """

    SUCCESS = 0
    P2_NONCONVEX = 1
    P2_FALLBACK3 = 2
    P3_BAD_NORMAL = 3
    P3_INVALID_V4 = 4
    P3_INVALID_V5 = 5
    P3_MISSING_ORIGIN = 6
    P3_ORIGIN_ON_FACE = 7
    P4_MISSING_ORIGIN = 8
    P4_FALLBACK3 = 9
