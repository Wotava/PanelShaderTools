from math import pi, cos, sin
from struct import pack, unpack
from heapq import heapreplace
import random
verbose = 0

def clamp(val, bottom_limit, top_limit):
    return max(bottom_limit, min(val, top_limit))

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


def encode_int_by_rule(value: float, rule: {}) -> int:
    if rule["raw"]:
        return clamp(value, rule["min_value"], rule["max_value"])
    else:
        return map_float_to_int_range(value, rule["min_value"], rule["max_value"], rule["bits"])


def encode_by_rule(values: [], target_ruleset: {}) -> int:
    channels = []

    active_channel = 0
    bits_cumulative = 0
    for value in values:
        pass


TEST_list = [
    [101,   32, "Distance", "float"],
    [111,   16, "Remap", "float"],
    [1,     16, "Offset", "float"],
    [76,    12, "DecalThickness", "float"],
    [32,    32, "2D Position.x", "float2"],
    [12,    32, "2D Position.y", "float2"],
    [22,    24, "Normal_yaw", "float"],
    [2123,  24, "Normal_pitch", "float"],
    [54,   6,  "Divs", "int"],
    [54,    12, "AngleOffset", "float"],
    [22,    12, "RemapAngular", "float"],
    [1, 1, "UseFG", "bool"],
    [20, 6, "FGSectors", "int"],
    [12, 6, "BGSectors", "int"],
    [5, 7, "SectorOffset", "int"],
    [1, 4, "PanelType", "int"],
    [15, 4, "flip_p1", "bool"],
    [15, 4, "flip_p2", "bool"],
]


def randomize_test():
    for i, val in enumerate(TEST_list):
        ceil = (2**val[1]) - 1
        new_val = int(random.random() * ceil)
        TEST_list[i][0] = new_val


def sublist_creator(values, splits):
    # Based on  https://stackoverflow.com/a/61649667
    # and       https://stackoverflow.com/a/613218
    bins = [[0] for _ in range(splits)]
    values = sorted(values, reverse=True, key=lambda item: item[1])

    # least[0] holds sum of all values in a "bin"
    for i in values:
        bit = i[1]
        # check if smallest bin is above 32 limit
        if bins[0][0] + bit > 32:
            raise ValueError('Cant pack stuff effectively at all')
        least = bins[0]
        least[0] += bit
        least.append(i)
        heapreplace(bins, least)

    return [x[1:] for x in bins]


def bool_list_to_mask(b_list: []) -> int:
    i = 0
    for val in b_list:
        i = i << 1 | int(val)
    return i


def ultra_generic_packer(values: [], validate=False) -> [int]:
    prepacked_channels = sublist_creator(values, 8)
    packed_channels = [0] * 8

    # pack values
    for i, channel in enumerate(prepacked_channels):
        packed = 0
        bits_sum = 0
        for pair in channel:
            value, bit, name = pair
            if name == 'flip_p1':
                p1_target = i
            elif name == 'flip_p2':
                p2_target = i
            bits_sum += bit
            packed = packed << bit | value
        packed_channels[i] = packed

    flips = [check_mask(val) for val in packed_channels]
    for i, flip in enumerate(flips):
        if flip:
            packed_channels[i] ^= (1 << 30)
    flip_p1 = flips[0:4]
    flip_p2 = flips[4:]
    packed_channels[p1_target] = ((packed_channels[p1_target] >> 4) << 4) | bool_list_to_mask(flip_p1)
    packed_channels[p2_target] = ((packed_channels[p2_target] >> 4) << 4) | bool_list_to_mask(flip_p2)

    # call validator
    if validate:
        validate_generic_pack(packed_channels, prepacked_channels)
    return packed_channels


def validate_generic_pack(packed_values: [], original_values: []) -> str:
    channel_names = ['color1.x', 'color1.y', 'color1.z', 'color1a', 'color2.x', 'color2.y', 'color2.z', 'color2a']
    flip_names = ['flip_r', 'flip_g', 'flip_b', 'flip_a']
    code = ""

    # unpack flip-flags (works)
    p1_target, p2_target = None, None
    for index, channel in enumerate(original_values):
        for pair in channel:
            value, bit, name = pair
            if name == 'flip_p1':
                p1_target = index
                break
            elif name == 'flip_p2':
                p2_target = index
                break
    flips = []
    if not p1_target or not p2_target:
        raise RuntimeError("Flip flags were not found!")
    targ = [p1_target, p2_target]
    for n in range(2):
        val = packed_values[targ[n]]
        flips_l = []
        for i in range(4):
            # get bitflag
            flips_l.append(bool(val & 1))
            val >>= 1

            # write the rule to the code
            code += (flip_names[i]+str(n+1)) + " = " + channel_names[targ[n]] + " & 1; \n"
            code += channel_names[targ[n]] + " >>= 1; \n"
        code += "\n"

        flips_l.reverse()
        flips.extend(flips_l)

    # add flip code to output
    for n in range(2):
        for i in range(4):
            code += f"if ({flip_names[i]+str(n+1)})" + " { \n"
            code += f"    {channel_names[i + (4*n)]} ^= (1 << 30); \n"
            code += "}; \n"
    code += "\n"

    for index, flip in enumerate(flips):
        if flip:
            packed_values[index] ^= (1 << 30)

    # unpack values
    for index, channel in enumerate(original_values):
        packed_channel = packed_values[index]
        for block in reversed(channel):
            value, bit, name = block
            if name[0:5] != "flip":
                test_value = packed_channel & (2**bit - 1)
                if test_value == value:
                    print(f"{name} matches")
                else:
                    print(f"{name} doesn't match, {value} != {test_value}")

                # add code
                code += f"{name} = {channel_names[index]} & (pow(2, {bit}) - 1); \n"
                code += f"{channel_names[index]} >>= {bit}; \n"

            packed_channel >>= bit

    print(code)

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
