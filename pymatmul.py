import subprocess
import sys
from importlib.util import find_spec

if not find_spec("numpy"):
    print("Numpy not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

from timeit import timeit

import numpy as np


class PyMatrix:
    def __init__(self, value, rows, cols):
        self.value = value
        self.rows = rows
        self.cols = cols

    def __getitem__(self, idxs):
        return self.value[idxs[0]][idxs[1]]

    def __setitem__(self, idxs, value):
        self.value[idxs[0]][idxs[1]] = value


def matmul_python(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


def benchmark_matmul_python(M, N, K):
    A = PyMatrix(list(np.random.rand(M, K)), M, K)
    B = PyMatrix(list(np.random.rand(K, N)), K, N)
    C = PyMatrix(list(np.zeros((M, N))), M, N)
    secs = timeit(lambda: matmul_python(C, A, B), number=2) / 2
    gflops = ((2 * M * N * K) / secs) / 1e9
    return gflops


if __name__ == "__main__":
    print("Throughput of a 128x128 matrix multiplication in Python:")
    benchmark_matmul_python(128, 128, 128)