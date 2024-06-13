"""
3D Morton code spatial hashing and Z-order curve bitwise interleaving.
"""

from src.math.vector import Vec3f

def expand_bits_10_to_30(val: UInt32) -> UInt32:
    """
    Expand 10-bit integer to 30 bits by inserting two zeros between adjacent bits:
    v = ---- ---- ---- ---- ---- --98 7654 3210
    -> --9- -8-- 7--6 --5- -4-- 3--2 --1- -0--
    """
    var v = val & 0x3FF # 10 bits: [0, 1023]
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8))  & 0x0300F00F
    v = (v | (v << 4))  & 0x030C30C3
    v = (v | (v << 2))  & 0x09249249
    return v


def compute_morton_3d(normalized_pos: Vec3f) -> UInt32:
    """
    Map normalized 3D position in [0, 1]^3 to 30-bit Morton code.
    """
    var nx = normalized_pos.x
    var ny = normalized_pos.y
    var nz = normalized_pos.z

    if nx < 0.0: nx = 0.0
    elif nx > 1.0: nx = 1.0

    if ny < 0.0: ny = 0.0
    elif ny > 1.0: ny = 1.0

    if nz < 0.0: nz = 0.0
    elif nz > 1.0: nz = 1.0

    var x_int = UInt32(nx * 1023.0)
    var y_int = UInt32(ny * 1023.0)
    var z_int = UInt32(nz * 1023.0)

    var xx = expand_bits_10_to_30(x_int)
    var yy = expand_bits_10_to_30(y_int)
    var zz = expand_bits_10_to_30(z_int)

    return (xx | (yy << 1) | (zz << 2))
