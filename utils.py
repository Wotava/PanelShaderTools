from math import asin, atan2, cos, sin, pi
from mathutils import Vector

verbose = 0


def get_rotator(vec: Vector, limit_range=True) -> [float]:
    """Returns Yaw and Pitch rotation in radians. Negates input Vec with
    negative Z and returns rotation values that match a collinear vector
    to keep values inside [-Pi/2, Pi/2] range"""

    if vec.length < 0.9999 or vec.length > 1.00001:
        print(f"[Vec to rot]Got non-normalized vector {vec.length}")

    if limit_range and vec.z < 0:
        vec = vec.copy()
        vec.negate()

    # asin will output values in range [-Pi/2, Pi/2]
    pitch = asin(-1 * vec.y)

    # atan2 returns values inside [-Pi, Pi] range when vec.z can be negative,
    # so we negate all vectors with negative z to reduce range to [-Pi/2, Pi/2]
    yaw = atan2(vec.x, vec.z)

    if verbose > 0:
        print(f"[GetRotator]Encoded vector: {vec}")
    return [yaw * -1, pitch * -1]


def restore_from_rotator(rotator: [float]) -> Vector:
    vec = Vector((0.0, 0.0, 0.0))
    yaw = rotator[0]
    pitch = rotator[1]
    xz_len = cos(pitch)
    vec.x = xz_len * sin(-yaw)
    vec.y = sin(pitch)
    vec.z = xz_len * cos(yaw)
    return vec
