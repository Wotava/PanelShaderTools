#!/usr/bin/env python
# This little script is supposed to convert data types from HLSL to GLSL ones and
# strip all UE LWC functions to make this code compatible with blender

import sys


def update_data_types(line: str) -> str:
    # HLSL default?
    line = line.replace("float3", "vec3")
    line = line.replace("float2", "vec2")

    # LWC
    line = line.replace("FLWCVector3", "vec3")
    line = line.replace("FLWCVector2", "vec2")
    line = line.replace("FLWCScalar", "float")

    line = line.replace("Vector3", "vec3")
    return line


def update_function_inputs(line: str) -> str:
    line = line.replace("	const FMaterialPixelParameters Parameters,",
                        "	const vec3 InPixelNormal, \n	const vec3 InPixelPosition, \n")
    return line


def replace_defines(line: str) -> str:
    line = line.replace("LWC_WS_POSITION", "InPixelPosition")
    line = line.replace("WS_NORMAL", "InPixelNormal")
    return line


def replace_std_functions(line: str) -> str:
    line = line.replace("fmod", "mod")

    if line.find("saturate") != -1:
        args = get_function_args(line, "saturate")
        args_adj = args + ", 0, 1"
        line = line.replace(f"saturate({args})", f"clamp({args_adj})")

    line = line.replace("LWCLength", "length")
    line = line.replace("LWCFmod", "mod")
    line = line.replace("LWCFloor", "floor")
    line = line.replace("LWCDot", "dot")
    line = line.replace("LWCCross", "cross")
    line = line.replace("MakeLWCVector", "vec3")
    return line


def strip_float_markers(line: str) -> str:
    for num in range(10):
        line = line.replace(f"{num}.f", str(num))
        line = line.replace(f"{num}f", str(num))
    return line


def get_function_args(line: str, function: str, omit_braces=True) -> str:
    start_pos = line.find(function)
    l_br, r_br = 0, 0
    target_l_br = None
    i = start_pos

    while i < len(line):
        symbol = line[i]
        if symbol == "(":
            if not target_l_br:
                target_l_br = i
            l_br += 1
        if symbol == ")":
            r_br += 1
        if l_br == r_br != 0:
            if not target_l_br:
                raise Exception(f"Function {function} in {line} has non-closed right brackets?")

            if omit_braces:
                return line[target_l_br + 1: i]
            else:
                return line[target_l_br: i + 1]
        i += 1
    raise Exception(f"Function {function} in {line} has non-closed brackets?")


def get_function_sub_args(line: str) -> [str]:
    sub_args = []
    l_br, r_br = 0, 0
    start = 0
    i = 0
    while i < len(line):
        symbol = line[i]

        if l_br == r_br and symbol == ',':
            sub_args.append(line[start:i])
            sub_args.append(symbol)
            l_br, r_br = 0, 0
            start = i + 1
            i += 1
            continue

        if symbol == "(":
            l_br += 1
        if symbol == ")":
            r_br += 1

        i += 1
    if start < i:
        sub_args.append(line[start:i+1])
    return sub_args


def split_to_function_and_arg(line: str) -> [str]:
    pass


def replace_lwc_functions(line: str, verbose=False) -> str:
    functions = [
        ["LWCToFloat", ""],
        ["LWCPromote", ""],
        ["LWCSubtract", " - "],
        ["LWCAdd", " + "],
        ["LWCDivide", " / "],
        ["LWCMultiply", " * "],
    ]
    #
    for function in functions:
        while line.find(function[0]) != -1:
            args = get_function_args(line, function[0])

            if verbose:
                print(function[0], "args: ", args)

            if function[1] != "":
                args_adj = get_function_sub_args(args)

                if verbose:
                    print(f"subargs: {args_adj}")

                if len(args_adj) > 1:
                    for i, sub_arg in enumerate(args_adj):
                        if sub_arg.replace(" ", "") == ',':
                            args_adj[i] = function[1]
                            break
                    args_adj = "(" + ''.join(args_adj) + ")"
                    if verbose:
                        print(f"complex replace res {args_adj}")

                else:
                    if verbose:
                        print("simple replace")
                    args_adj = args.replace(",", function[1])
            else:
                args_adj = args
            line = line.replace(f"{function[0]}({args})", args_adj)
            if verbose:
                print(f"res {line} \n")
    return line


with open(sys.argv[1], 'r') as my_file:
    s = my_file.read()
    lines = s.split(";")
    for i in range(len(lines)):
        lines[i] += ';'

    for sub_line in lines:
        sub_line = update_data_types(sub_line)
        sub_line = update_function_inputs(sub_line)
        sub_line = replace_defines(sub_line)
        sub_line = replace_std_functions(sub_line)
        sub_line = strip_float_markers(sub_line)
        sub_line = replace_lwc_functions(sub_line)
        print(sub_line, end='')
