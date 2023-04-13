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


def pack_manual(target: [float]):
    packed = target[0] << 24 | target[1] << 16 | target[2] << 8 | target[3]
    return packed


def unpack_manual(target: int) -> [int]:
    d = target & 255
    abc = target >> 8
    c = abc & 255
    ab = abc >> 8
    b = ab & 255
    a = ab >> 8
    return a, b, c, d
