"""Day 19 / Session 2 - Lab Assignment 1: Freeze the Backbone - Trainable vs Frozen Parameter Budget."""

backbone = [
    {"type": "conv", "kh": 3, "kw": 3, "in_ch": 3, "out_ch": 32},
    {"type": "pool"},
    {"type": "conv", "kh": 3, "kw": 3, "in_ch": 32, "out_ch": 64},
    {"type": "pool"},
    {"type": "conv", "kh": 3, "kw": 3, "in_ch": 64, "out_ch": 128},
    {"type": "pool"},
    {"type": "conv", "kh": 3, "kw": 3, "in_ch": 128, "out_ch": 128},
    {"type": "pool"},
]
head = [
    {"type": "flatten"},
    {"type": "dense", "in": 128, "out": 128},
    {"type": "dense", "in": 128, "out": 10},
]


def layer_params(layer):
    kind = layer.get("type")
    if kind == "conv":
        return (layer["kh"] * layer["kw"] * layer["in_ch"] + 1) * layer["out_ch"]   # +1 bias per filter
    if kind == "dense":
        return (layer["in"] + 1) * layer["out"]                                     # +1 bias per unit
    return 0                                                                        # pool / flatten / relu / dropout


def count_params(layers):
    return sum(layer_params(layer) for layer in layers)


def build_transfer_model(backbone, head, freeze_backbone=True):
    back, new = count_params(backbone), count_params(head)
    total = back + new
    if freeze_backbone:
        return {"trainable": new, "frozen": back, "total": total}
    return {"trainable": total, "frozen": 0, "total": total}


def trainable_ratio(model):
    if model["total"] == 0:
        return 0.0
    return model["trainable"] / model["total"]


def finetune_topk(backbone, head, k):
    """Unfreeze the last k parameter-bearing backbone layers (nearest the head) plus the whole head."""
    total = count_params(backbone) + count_params(head)
    bearing = [layer_params(layer) for layer in backbone if layer_params(layer) > 0]
    if k <= 0:
        thawed = 0
    else:
        thawed = sum(bearing[-k:])               # k larger than the list simply takes everything
    trainable = count_params(head) + thawed
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


def strategy_budgets(backbone, head):
    return {
        "feature_extraction": build_transfer_model(backbone, head, True)["trainable"],
        "finetune_topk2": finetune_topk(backbone, head, 2)["trainable"],
        "full": build_transfer_model(backbone, head, False)["trainable"],
    }


if __name__ == "__main__":
    m = build_transfer_model(backbone, head, freeze_backbone=True)
    print(m)                                              # {'trainable': 17802, 'frozen': 240832, 'total': 258634}
    print(round(trainable_ratio(m), 3))                   # 0.069
    print(build_transfer_model(backbone, head, False))    # {'trainable': 258634, 'frozen': 0, 'total': 258634}
    print(finetune_topk(backbone, head, 1))               # {'trainable': 165386, 'frozen': 93248, 'total': 258634}
    print(strategy_budgets(backbone, head))               # {'feature_extraction': 17802, 'finetune_topk2': 239242, 'full': 258634}
    print(layer_params(backbone[0]), layer_params(head[2]), layer_params({"type": "pool"}), count_params(backbone))  # 896 1290 0 240832
    print(finetune_topk(backbone, head, 0)["trainable"], finetune_topk(backbone, head, 99)["trainable"])             # 17802 258634
