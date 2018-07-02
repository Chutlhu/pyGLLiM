import numpy as np
import time

# typical values
D = 10
Lt = 2
Lw = 1
N = 2000

times = []
times_mkl = []


def foo(t, y, D, Lt, Lw, N):

    Atk = np.random.rand(D, Lt) # DxLt
    Awk = np.random.rand(D, Lw) # DxLw
    bk = np.random.rand(D,1) # Dx1
    Σk = np.random.rand(D, D) # DxD
    Σk = np.matmul(Σk,Σk.T)
    Γwk = np.random.rand(Lw, Lw) # LwxLw
    Γwk = np.matmul(Γwk,Γwk.T)
    cwk = np.random.rand(Lw,1) # Lwx1

    invΓwk = np.linalg.inv(Γwk)
    invΣk = np.linalg.inv(Σk)
    invSwk = invΓwk + np.matmul(np.matmul(Awk.T, invΣk), Awk) # LwxLw

    Atkt = np.dot(Atk, t) # DxLt

    Sw = np.linalg.inv(invSwk)
    μw = np.matmul(Sw,
                    np.matmul(np.matmul(Awk.T, invΣk),
                        y - Atkt - bk) \
                    + np.dot(invΓwk, cwk)) # LwxN
    return

for D in [10, 100, 200]:
    t = np.random.rand(Lt,N)
    y = np.random.rand(D,N)
    start = time.time()
    foo(t, y, D, Lt, Lw, N)
    times.append(time.time())


for D in [10, 100, 200]:
    t = np.random.rand(Lt,N)
    y = np.random.rand(D,N)
    start = time.time()
    foo(t, y, D, Lt, Lw, N)
    times_mkl.append(time.time())


print(min(times), min(times_mkl))
