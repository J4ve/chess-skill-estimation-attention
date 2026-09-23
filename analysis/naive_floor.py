"""Naive predict-a-constant floor on the held-out test partition.

Reads the per-game test file already written by the held-out evaluation
(analysis/heldout_test_eval/baseline__best.csv, one row per test game with both
players' true ratings) and reports the mean absolute error of predicting one
constant rating for every player, in the same per-game-then-averaged form as the
model MAE of Chapter 4.

It first reproduces the reported baseline MAE from the same file, so the two
numbers are known to be measured the same way.

Usage:  python analysis/scripts/naive_floor.py
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "heldout_test_eval", "baseline__best.csv")

white, black, w_err, b_err = [], [], [], []
with open(CSV) as f:
    for r in csv.DictReader(f):
        white.append(float(r["white_elo"]))
        black.append(float(r["black_elo"]))
        w_err.append(float(r["white_err"]))
        b_err.append(float(r["black_err"]))

n = len(white)
model_mae = sum((a + b) / 2 for a, b in zip(w_err, b_err)) / n


def floor(c):
    return sum((abs(a - c) + abs(b - c)) / 2 for a, b in zip(white, black)) / n


ratings = white + black
mean, median = sum(ratings) / len(ratings), st.median(ratings)
print(f"test games                          {n}")
print(f"baseline MAE recomputed from file   {model_mae:.4f}   (Chapter 4 reports 175.00)")
print(f"test-partition mean rating          {mean:.2f}")
print(f"predict-the-mean MAE                {floor(mean):.2f}")
print(f"predict-the-median MAE              {floor(median):.2f}")
