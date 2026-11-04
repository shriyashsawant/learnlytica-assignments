"""Day 18 / Session 2 - Lab Assignment 1: The Parameter Budget - count a CNN's weights the VGG way."""

# (kernel, c_in, c_out) triples describing real VGG-16 conv layers
CONV_LAYERS = [(3, 3, 64), (3, 64, 64), (3, 64, 128), (3, 128, 128)]
# (n_in, n_out) for VGG-16's fully-connected head
FC_LAYERS = [(25088, 4096), (4096, 4096), (4096, 1000)]


def conv_params(k, c_in, c_out, bias=True):
    """Each of the c_out output channels owns one k*k*c_in kernel, plus one bias per output channel."""
    weights = k * k * c_in * c_out
    return weights + c_out if bias else weights


def dense_params(n_in, n_out, bias=True):
    weights = n_in * n_out
    return weights + n_out if bias else weights


def receptive_field(kernel_sizes):
    rf = 1                                    # a single pixel sees itself
    for k in kernel_sizes:
        rf += k - 1                           # each stride-1 k x k conv adds k-1
    return rf


def vgg_block_params(c_in, c_out, n_convs=2, k=3):
    total = conv_params(k, c_in, c_out)       # first conv: c_in -> c_out
    for _ in range(n_convs - 1):
        total += conv_params(k, c_out, c_out)  # the remaining convs: c_out -> c_out
    return total


def compare_5x5_vs_two_3x3(c):
    single = conv_params(5, c, c)
    two = 2 * conv_params(3, c, c)
    return {
        "single_5x5_params": single,
        "two_3x3_params": two,
        "single_5x5_rf": receptive_field([5]),
        "two_3x3_rf": receptive_field([3, 3]),
        "fewer_params": "two_3x3" if two < single else ("single_5x5" if single < two else "equal"),
    }


if __name__ == "__main__":
    print(conv_params(3, 3, 64), conv_params(3, 64, 64, bias=False), dense_params(25088, 4096))
    print(receptive_field([3, 3]), receptive_field([5]), receptive_field([3, 3, 3]))
    print(vgg_block_params(3, 64, 2, 3), vgg_block_params(64, 128, 2, 3), vgg_block_params(16, 32, 3, 3))
    print(compare_5x5_vs_two_3x3(64))
