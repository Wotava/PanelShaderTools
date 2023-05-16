from math import pi, cos, sin
from struct import pack, unpack
verbose = 0


def map_float_to_int_range(value: float, range_min: float, range_max: float, bit_depth: int) -> int:
    int_max = pow(2, bit_depth) - 1
    dist_from_min = value - range_min
    value_range = range_max - range_min
    return int(dist_from_min / value_range * int_max)


def map_int_to_float_range(value: float, range_min: int, range_max: int, bit_depth: int) -> float:
    val_range = (range_max - range_min)
    max_int = pow(2, bit_depth) - 1
    multiplier = value / max_int
    return float(range_min + val_range * multiplier)


def as_float(target: int, endian='big') -> float:
    if endian == 'big':
        fmt_i = '>I'
        fmt_f = '>f'
    else:
        fmt_i = '<I'
        fmt_f = '<f'
    bytepres = pack(fmt_i, target)
    return unpack(fmt_f, bytepres)[0]


def check_mask(target: int) -> bool:
    """
    Returns True if this int value will not work with Blender due to value clamping or rounding.
    Exponent bits should never start with 1 like 1xxxxxxx, exception is 10000000 (because 00000000 won't work too).
    """
    if target == 0:
        return False
    t_bin = bin(target)
    t_bin = ('0' * (34 - len(t_bin))) + t_bin[2:]
    if t_bin[1:9] == '00000000' or (t_bin[1:9] != '10000000' and t_bin[1] == '1'):
        return True
    else:
        return False


def as_float_denormalized(target: float) -> [float, bool]:
    """A somewhat lazy way to surpass Blender's ImageTexture node output clamp of extremely large float values by
    de-normalizing said float, so the output will be in range [-2, 2], which is inside Blender's clamp range.
    Outputs de-normalized (30th bit set to 0) float value and a bool flag that is set to true when original value
    had 30th bit set to 1"""
    if type(target) is float:
        target = as_uint(target)

    if check_mask(target):
        target ^= (1 << 30)
        return [as_float(target, 'big'), True]
    else:
        return [as_float(target, 'big'), False]


def as_uint(target: float, endian='big') -> int:
    if endian == 'big':
        fmt_i = '>I'
        fmt_f = '>f'
    else:
        fmt_i = '<I'
        fmt_f = '<f'
    bytepres = pack(fmt_f, target)
    return unpack(fmt_i, bytepres)[0]


def pack_manual(target: [int], bits_per_val: [int]) -> int:
    """Pack multiple int values with varying length into one int32 value with bitwise operations"""
    if type(bits_per_val) is int:
        bits_per_val = [bits_per_val]
    if len(bits_per_val) == 1:
        bits_per_val = bits_per_val * len(target)
    if len(bits_per_val) != 1 and len(bits_per_val) != len(target):
        raise ValueError("[Byte encode]Amount of bit-lengths provided doesn't match amount of target values")
    if sum(bits_per_val) > 32:
        raise OverflowError("[Byte encode]Sum of provided bit-lengths exceeds 32")

    packed = 0
    for val_i, val in enumerate(target):
        # Check if values fit inside provided byte lengths
        if val > 2**(bits_per_val[val_i]) - 1:
            print(f"[Byte encode]Warning: value {val} exceeds maximum value of {2 ** (bits_per_val[val_i]) - 1} "
                  f"at {bits_per_val[val_i]} bits length, output will be broken")
        if verbose > 1:
            print(f"{val_i}: {packed} = {packed} << {bits_per_val[val_i]} | {val} ")
        packed = packed << bits_per_val[val_i] | val
    return packed


def unpack_manual(target: int, bits_per_val: [int]) -> [int]:
    """Unpack multiple int values with varying lengths from one int32 value with bitwise operations"""
    result = []
    # Values are being unpacked in reverse order, so we reverse both the result AND the bite lengths list
    bits_per_val = bits_per_val.copy()
    bits_per_val.reverse()
    for bit_length in bits_per_val:
        if verbose > 1:
            print(f"value {len(result)}: {target} & {2**bit_length - 1}")
            print(f"target = {target} >> {bit_length}")
        result.append(target & (2**bit_length - 1))
        target = target >> bit_length
    result.reverse()
    return result


def rotator_unpack_test(target: float):
    r_int = as_uint(target)

    pitch = map_int_to_float_range(float(r_int & 32767), 0, 1, 15) * (pi / 2)
    r_int = r_int >> 15
    pitch *= 1 - ((r_int & 1) * 2)
    r_int = r_int >> 1

    yaw = map_int_to_float_range(float(r_int & 32767), 0, 1, 15) * (pi / 2)
    r_int = r_int >> 15
    yaw *= 1 - ((r_int & 1) * 2)

    xz_len = cos(pitch)
    x = xz_len * sin(-yaw)
    y = sin(pitch)
    z = xz_len * cos(yaw)
    return [x, y, z]
