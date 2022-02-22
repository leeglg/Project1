# 1_3
import numpy as np

def leastsquaresol():
    # A가 정방행렬 아닐 때
    # A B 만들고
    tmp1 = np.random.randint(1, 5)
    tmp2 = np.random.randint(1, 5)

    A = np.random.randint(0, 10, size=(tmp1, tmp2))
    B = np.random.randint(0, 10, size=(tmp2, 1))
    print("A\n", A)
    print("B\n", B)

    # A 유사 역행렬 구해서
    Apinv = np.linalg.pinv(A)
    print("Ainv\n", Apinv)

    # x = A^-1 * B
    x = np.einsum("ij,jk->ik", Apinv, B)
    print("x\n", x)