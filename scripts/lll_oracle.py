#!/usr/bin/env python3
"""Independent LLL oracle for rung D.

This is the textbook *rational* Gram-Schmidt LLL (Cohen 2.6.1 in spirit):
mu and the squared norms B_i are exact Fractions, computed from a fresh
Gram-Schmidt each pass. It shares no code with rump's integer-d recurrence
(Cohen 2.6.3), so agreement between the two is a real cross-check rather
than two copies of one bug.

Convention: delta = 3/4 default; nearest integer is floor(x + 1/2)
(ties round toward +inf), matching Cohen's <x> = floor(x + 1/2).
"""
from fractions import Fraction as F
import math


def dot(u, v):
    return sum(F(a) * F(b) for a, b in zip(u, v))


def gram_schmidt(b):
    n = len(b)
    bstar = [[F(x) for x in row] for row in b]
    mu = [[F(0)] * n for _ in range(n)]
    B = [F(0)] * n
    for i in range(n):
        bstar[i] = [F(x) for x in b[i]]
        for j in range(i):
            mu[i][j] = dot(b[i], bstar[j]) / B[j]
            bstar[i] = [bstar[i][k] - mu[i][j] * bstar[j][k] for k in range(len(b[i]))]
        B[i] = dot(bstar[i], bstar[i])
    return bstar, mu, B


def rnd(x):
    # floor(x + 1/2), exact on Fractions
    return math.floor(x + F(1, 2))


def lll(b, delta=F(3, 4)):
    b = [list(row) for row in b]
    n = len(b)
    k = 1
    while k < n:
        bstar, mu, B = gram_schmidt(b)
        # size-reduce b[k] against b[k-1]
        if abs(mu[k][k - 1]) > F(1, 2):
            q = rnd(mu[k][k - 1])
            b[k] = [b[k][i] - q * b[k - 1][i] for i in range(len(b[k]))]
            bstar, mu, B = gram_schmidt(b)
        # Lovasz condition
        if B[k] >= (delta - mu[k][k - 1] ** 2) * B[k - 1]:
            # full size reduction of b[k] against b[k-2..0]
            for l in range(k - 2, -1, -1):
                if abs(mu[k][l]) > F(1, 2):
                    q = rnd(mu[k][l])
                    b[k] = [b[k][i] - q * b[l][i] for i in range(len(b[k]))]
                    bstar, mu, B = gram_schmidt(b)
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            k = max(k - 1, 1)
    return b


def gram_det(b):
    # det of the Gram matrix b b^T (lattice invariant, = (covolume)^2)
    n = len(b)
    G = [[dot(b[i], b[j]) for j in range(n)] for i in range(n)]
    # fraction-free would do; Bareiss-lite via Fraction elimination
    G = [[F(x) for x in row] for row in G]
    det = F(1)
    for i in range(n):
        # pivot
        p = i
        while p < n and G[p][i] == 0:
            p += 1
        if p == n:
            return F(0)
        if p != i:
            G[i], G[p] = G[p], G[i]
            det = -det
        det *= G[i][i]
        for r in range(i + 1, n):
            f = G[r][i] / G[i][i]
            for c in range(i, n):
                G[r][c] -= f * G[i][c]
    return det


def is_size_reduced(b):
    _, mu, _ = gram_schmidt(b)
    for i in range(len(b)):
        for j in range(i):
            if abs(mu[i][j]) > F(1, 2):
                return False
    return True


def satisfies_lovasz(b, delta=F(3, 4)):
    _, mu, B = gram_schmidt(b)
    for k in range(1, len(b)):
        if B[k] < (delta - mu[k][k - 1] ** 2) * B[k - 1]:
            return False
    return True


TESTS = {
    # Cohen-style small integer lattices, plus adversarial ones.
    "eye3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "cohen_ex": [[1, 1, 1], [-1, 0, 2], [3, 5, 6]],
    "skew": [[201, 37], [1648, 297]],   # 2D, Gauss reduction
    "hard4": [[1, 0, 0, 1345],
              [0, 1, 0, 3571],
              [0, 0, 1, 8765],
              [0, 0, 0, 10007]],  # knapsack-style, forces long reduction
    "neg": [[-2, 7, 3], [5, -1, 4], [0, 6, -8]],
    "collinear_free": [[2, 4], [3, 1]],
    "big": [[123456789, 0, 0],
            [0, 987654321, 0],
            [111111111, 222222222, 333333333]],
}

for name, basis in TESTS.items():
    red = lll(basis)
    d0 = gram_det(basis)
    d1 = gram_det(red)
    ok_sr = is_size_reduced(red)
    ok_lv = satisfies_lovasz(red)
    assert d0 == d1, f"{name}: gram det changed {d0} -> {d1}"
    assert ok_sr and ok_lv, f"{name}: not reduced sr={ok_sr} lv={ok_lv}"
    print(f"{name}:")
    print(f"    input   = {basis}")
    print(f"    reduced = {red}")
    print(f"    gram_det= {d0}  size_reduced={ok_sr} lovasz={ok_lv}")
print("ALL ORACLE CASES PASS")
