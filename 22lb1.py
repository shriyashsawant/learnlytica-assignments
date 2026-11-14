"""Day 22 / Session 1 - Lab Assignment 1: Teacher Forcing - Shift the Target to Build Decoder Inputs & Targets."""
import numpy as np


def decoder_inputs(target, start_token=1):
    target = np.asarray(target, dtype=int)
    return np.concatenate([[start_token], target[:-1]]).astype(int)       # [START] + target[:-1]: same length as target


def decoder_targets(target):
    return np.asarray(target, dtype=int)                                   # the labels, unchanged


def teacher_forcing(target, start_token=1):
    return decoder_inputs(target, start_token), decoder_targets(target)


if __name__ == "__main__":
    print(decoder_inputs([7, 8, 9, 2]).tolist())                           # [1, 7, 8, 9]
    print(decoder_targets([7, 8, 9, 2]).tolist())                          # [7, 8, 9, 2]
    di, do = teacher_forcing([7, 8, 9, 2])
    print(len(di) == len(do) == 4)                                         # True
    print(di[1:].tolist() == do[:-1].tolist())                             # True
    print(decoder_inputs([5, 6, 2], start_token=9).tolist())               # [9, 5, 6]
