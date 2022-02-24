# 1_3
import numpy as np


def nsel(m):
    n = np.random.randint(1, 3)
    if (n==m):
        return nsel(m)
    return n

def leastsquaresol():
    # A가 정방행렬 아닐 때
    # A B 만들고
    m = np.random.randint(1, 3) # A의 행 수 = B의 행 수
    n = nsel(m)                 # A의 열 수 =/= m


    tmp3 = np.random.randint(1, 3) # B의 열 수
    A = np.random.randint(0, 10, size=(m, n))
    B = np.random.randint(0, 10, size=(m, tmp3))

    print("A\n", A)
    print("B\n", B)

    # A 유사 역행렬 구해서
    Apinv = np.linalg.pinv(A)
    print("Ainv\n", Apinv)

    # x = A^-1 * B   근사값.
    x = np.einsum("ij,jk->ik", Apinv, B)
    print("x\n", x)