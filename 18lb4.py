"""Day 18 / Session 2 - Lab Assignment 2: Trace the Feature Map - spatial sizes and MAC/FLOP budgets."""

# a VGG block1 applied to a 224x224 image
VGG_BLOCK1 = [
    {"type": "conv", "k": 3, "stride": 1, "pad": 1},
    {"type": "conv", "k": 3, "stride": 1, "pad": 1},
    {"type": "pool", "k": 2},
]


def conv_out_size(in_size, k, stride=1, padding=0):
    return (in_size + 2 * padding - k) // stride + 1


def pool_out_size(in_size, k, stride=None):
    if stride is None:
        stride = k                      # the usual non-overlapping pooling
    return (in_size - k) // stride + 1


def trace(in_size, layers):
    sizes = []
    size = in_size
    for layer in layers:
        if layer["type"] == "conv":
            size = conv_out_size(size, layer["k"], layer.get("stride", 1), layer.get("pad", 0))
        elif layer["type"] == "pool":
            size = pool_out_size(size, layer["k"], layer.get("stride"))
        else:
            raise ValueError("unknown layer type: " + str(layer["type"]))
        sizes.append(size)
    return sizes


def conv_macs(out_size, k, c_in, c_out):
    return out_size * out_size * c_out * (k * k * c_in)


if __name__ == "__main__":
    print(conv_out_size(224, 3, 1, 1), conv_out_size(32, 5, 1, 0), pool_out_size(224, 2), pool_out_size(28, 2))
    print(trace(224, VGG_BLOCK1))
    lenet = [{"type": "conv", "k": 5, "stride": 1, "pad": 0}, {"type": "pool", "k": 2},
             {"type": "conv", "k": 5, "stride": 1, "pad": 0}, {"type": "pool", "k": 2}]
    print(trace(32, lenet))
    print(conv_macs(224, 3, 3, 64))
    print(conv_out_size(224, 7, 2, 3), conv_out_size(7, 3, 2, 0))
    print(2 * conv_macs(28, 3, 64, 64) < conv_macs(28, 5, 64, 64))
