from struct import pack, unpack


def map_float_to_int8(target: float) -> int:
    return int(target * 255)


def map_int8_to_float(target: int) -> float:
    return target / 255


def map_pixel_to_int8(target: [float]) -> [int]:
    return [map_float_to_int8(val) for val in target]


def map_pixel_to_float(target: [int]) -> [float]:
    return [map_int8_to_float(val) for val in target]


def pack_struct(target: [float]):
    fmt = '=BBBB'
    bytepres = pack(fmt, *target)
    fmt = '=f'
    return unpack(fmt, bytepres)


def unpack_struct(target: float) -> [int]:
    fmt = '=f'
    bytepres = pack(fmt, target)
    print(unpack('=I', bytepres))
    fmt = '=BBBB'
    return unpack(fmt, bytepres)


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
        packed = packed << bits_per_val[val_i] | val
    return packed


def unpack_manual(target: int, bits_per_val: [int]) -> [int]:
    """Unpack multiple int values with varying lengths from one int32 value with bitwise operations"""
    result = []
    # Values are being unpacked in reverse order, so we reverse both the result AND the bite lengths list
    bits_per_val.reverse()
    for bit_length in bits_per_val:
        result.append(target & (2**bit_length - 1))
        target = target >> bit_length
    result.reverse()
    return result
