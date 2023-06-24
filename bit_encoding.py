from math import pi, cos, sin
from struct import pack, unpack
from heapq import heapreplace
import random
verbose = 0


def clamp(val, bottom_limit, top_limit):
    return max(bottom_limit, min(val, top_limit))


def sign_str(val) -> str:
    if val < 0:
        return "-"
    else:
        return ""


def pi_to_glsl_str(val: float) -> str:
    # I know this is stupid, but I'm lazy to change ranges right now
    if abs(val) == pi:
        return sign_str(val) + "M_PI"
    elif abs(val) == pi/2:
        return sign_str(val) + "M_PI/2"
    else:
        raise ValueError


def map_float_to_int_range(value: float, range_min: float, range_max: float, bit_depth: int) -> int:
    range_min_2 = 0
    range_max_2 = pow(2, bit_depth) - 1
    return int(range_min_2 + (value - range_min) * (range_max_2 - range_min_2) / (range_max - range_min))


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


def encode_by_rule(values: [], target_ruleset: []) -> [int]:
    res = [[]] * len(values)
    for i, pair in enumerate(values):
        value, name = pair
        rule = target_ruleset[name]
        if rule["raw"]:
            value = clamp(value, rule["min_value"], rule["max_value"])
        else:
            value = map_float_to_int_range(value, rule["min_value"], rule["max_value"], rule["bits"])
        # expected conversion structure: [value, bit length, name, rule]
        res[i] = [value, rule["bits"], name, rule]
    return res


TEST_list = [
    [234, "distance"],
    [0.54, "remap"],
    [0.645, "offset"],
    [2.2, "decal_thickness"],
    [-123.3, "2D_pos.x"],
    [45.0, "2D_pos.y"],
    [pi/4, "normal_yaw"],
    [-pi/7, "normal_pitch"],
    [43, "divs"],
    [0.54, "angle_offset"],
    [0.43, "remap_angular"],
    [1,  "use_fg"],
    [20, "fg_sectors"],
    [12, "bg_sectors"],
    [5, "sector_offset"],
    [1, "panel_type"],
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
    for index, val in enumerate(b_list):
        i = i | int(val) << (len(b_list) - 1 - index)
    return i


def ultra_generic_packer(values: [], validate=False) -> [int]:
    # expected prepacked structure: [value, name, rule{type, min, max, raw, bits}]
    values = values.copy()
    values.append([15, 4, "flip_p1", "bool"])
    values.append([15, 4, "flip_p2", "bool"])
    prepacked_channels = sublist_creator(values, 8)
    packed_channels = [0] * 8

    for i, channel in enumerate(prepacked_channels):
        for pair in channel:
            value, bit, name, rule = pair
            if name == 'flip_p1':
                p1_target = i
            elif name == 'flip_p2':
                p2_target = i

    if p1_target == p2_target:
        new_channel = []
        tail = [[], []]
        target_channel = prepacked_channels[p1_target]
        for v in target_channel:
            if v[2].find("flip") == -1:
                new_channel.append(v)
            else:
                if v[2] == "flip_p1":
                    tail[1] = v
                else:
                    tail[0] = v
        new_channel.extend(tail)
        prepacked_channels[p1_target] = new_channel
    else:
        t = [p1_target, p2_target]
        for n in range(2):
            new_channel = []
            tail = []
            target_channel = prepacked_channels[t[n]]
            for v in target_channel:
                if v[2].find("flip") == -1:
                    new_channel.append(v)
                else:
                    tail = v
            new_channel.append(tail)
            prepacked_channels[t[n]] = new_channel

    # pack values
    for i, channel in enumerate(prepacked_channels):
        packed = 0
        bits_sum = 0
        for pair in channel:
            value, bit, name, rule = pair
            bits_sum += bit
            packed = packed << bit | value
        packed_channels[i] = packed

    flips = [check_mask(val) for val in packed_channels]
    for i, flip in enumerate(flips):
        if flip:
            packed_channels[i] ^= (1 << 30)

    flip_p1 = flips[0:4]
    flip_p2 = flips[4:]
    # reverse flips before writing, so we will unpack in r/g/b/a from the end
    flip_p1.reverse()
    flip_p2.reverse()
    flips.reverse()

    if p1_target != p2_target:
        packed_channels[p1_target] = ((packed_channels[p1_target] >> 4) << 4) | bool_list_to_mask(flip_p1)
        packed_channels[p2_target] = ((packed_channels[p2_target] >> 4) << 4) | bool_list_to_mask(flip_p2)
    else:
        packed_channels[p1_target] = ((packed_channels[p1_target] >> 8) << 8) | bool_list_to_mask(flips)

    # call validator
    if validate:
        validate_generic_pack(packed_channels, prepacked_channels)
    return packed_channels


def validate_generic_pack(packed_values: [], original_values: [], print_code=False) -> str:
    channel_names = ['color1.x', 'color1.y', 'color1.z', 'color1a', 'color2.x', 'color2.y', 'color2.z', 'color2a']
    flip_names = ['flip_r', 'flip_g', 'flip_b', 'flip_a']
    declared_variables = []
    code = "//GENERATED CODE START \n"

    for channel in channel_names:
        code += f"{channel} = floatBitsToUint({channel});\n"

    # unpack flip-flags (works)
    p1_target, p2_target = None, None
    for index, channel in enumerate(original_values):
        for pair in channel:
            value, bit, name, rule = pair
            if name == 'flip_p1':
                p1_target = index
            elif name == 'flip_p2':
                p2_target = index
    flips = []
    if (p1_target is None) or (p2_target is None):
        raise RuntimeError("Flip flags were not found!")
    targ = [p1_target, p2_target]

    val = None
    for n in range(2):
        if not val or cval != packed_values[targ[n]]:
            val = packed_values[targ[n]]
            cval = val

        flips_l = []
        for i in range(4):
            # get bitflag
            flips_l.append(bool(val & 1))
            val >>= 1

            # write the rule to the code
            code += ("bool " + flip_names[i]+str(n+1)) + " = " + channel_names[targ[n]] + " & 1; \n"
            code += channel_names[targ[n]] + " >>= 1; \n"
        code += "\n"
        flips.extend(flips_l)

    # add flip code to output
    for n in range(2):
        for i in range(4):
            code += f"if ({flip_names[i]+str(n+1)})" + " { \n"
            code += f"    {channel_names[i + (4*n)]} ^= (1 << 30); \n"
            code += "} \n"
    code += "\n"

    for index, flip in enumerate(flips):
        if flip:
            packed_values[index] ^= (1 << 30)

    # unpack values
    for index, channel in enumerate(original_values):
        packed_channel = packed_values[index]
        for block in reversed(channel):
            value, bit, name, rule = block
            if name.find('flip') != -1:
                packed_channel >>= bit
                continue

            var_type = rule["type"]
            test_value = packed_channel & (2**bit - 1)
            if test_value == value:
                print(f"{name} matches")
            else:
                print(f"{name} doesn't match, {value} != {test_value}")
                print(f"->In channel {index}, flip status {flips[index]}")

            # add code
            # check declaration
            if name[-2] == ".":
                if name[:-2] not in declared_variables:
                    code += f"{var_type} {name[:-2]}; \n"
                    declared_variables.append(name[:-2])
                code += f"{name} = "
            else:
                if name not in declared_variables:
                    declared_variables.append(name)
                code += f"{var_type} {name} = "
            converted_value = f"{channel_names[index]} & (pow(2, {bit}) - 1)"

            if not rule["raw"]:
                mapped_value = f"map({converted_value}, 0, (pow(2, {bit}) - 1), "
                pi_test = [pi, pi/2]
                if abs(rule['min_value']) not in pi_test and abs(rule['max_value']) not in pi_test:
                    mapped_value += f"{rule['min_value']}.0, {rule['max_value']}.0)"
                else:
                    if abs(rule['min_value']) in pi_test:
                        mapped_value += f"{pi_to_glsl_str(rule['min_value'])}, "
                    else:
                        mapped_value += f"{rule['min_value']}, "

                    if abs(rule['max_value']) in pi_test:
                        mapped_value += f"{pi_to_glsl_str(rule['max_value'])})"
                    else:
                        mapped_value += f"{rule['max_value']})"
            else:
                mapped_value = converted_value

            if rule['type'] in ['bool', 'int']:
                mapped_value = f"{rule['type']}({mapped_value}); \n"
            else:
                mapped_value += "; \n"
            code += mapped_value

            code += f"{channel_names[index]} >>= {bit}; \n"
            code += "\n"

            packed_channel >>= bit

    code += "//GENERATED CODE END \n"
    if print_code:
        print(code)
    return code


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
