"""Day 22 / Session 1 - Lab Assignment 1: Teacher Forcing - Shift the Target to Build Decoder Inputs & Targets."""
import numpy as np


def make_decoder_io(target, bos_id=1, eos_id=2):
    target = list(target)
    return {"input": [bos_id] + target, "target": target + [eos_id]}      # the teacher-forcing shift


def pad(seq, length, pad_id=0):
    seq = list(seq)
    if len(seq) >= length:
        return seq[:length]                                              # truncate
    return seq + [pad_id] * (length - len(seq))                          # right-pad


def loss_mask(dec_target, pad_id=0):
    return (np.asarray(dec_target) != pad_id).astype(float)             # 1.0 on real tokens, 0.0 on PAD


def build_batch(targets, bos_id=1, eos_id=2, pad_id=0):
    ios = [make_decoder_io(t, bos_id, eos_id) for t in targets]
    L = max(len(io["input"]) for io in ios)                              # max(len(target) + 1)
    dec_in = np.array([pad(io["input"], L, pad_id) for io in ios])
    dec_out = np.array([pad(io["target"], L, pad_id) for io in ios])
    mask = loss_mask(dec_out, pad_id)
    return dec_in, dec_out, mask


if __name__ == "__main__":
    targets = [[7, 8, 9], [4, 5], [3, 6, 6, 7]]
    print(make_decoder_io([7, 8, 9]))                          # {'input': [1, 7, 8, 9], 'target': [7, 8, 9, 2]}
    di, do, m = build_batch(targets)
    print(di.shape, do.shape)                                  # (3, 5) (3, 5)
    print(do[1].tolist(), m[1].tolist())                       # [4, 5, 2, 0, 0] [1.0, 1.0, 1.0, 0.0, 0.0]
    print(m.sum(axis=1).tolist())                              # [4.0, 3.0, 5.0]
    print(pad([1, 2], 4), pad([1, 2, 3, 4, 5], 3))             # [1, 2, 0, 0] [1, 2, 3]
    print([di[i, 0] for i in range(3)], all(di[i, 1:len(targets[i]) + 1].tolist() == do[i, :len(targets[i])].tolist() for i in range(3)))   # [1, 1, 1] True
