# 1_3
import numpy as np

def linearsol(n):
    # A가 정방행렬일 때
    # A B 만들고



    A = np.random.randint(0, 10, size=(n, n))
    B = np.random.randint(0, 10, size=(n, 1))
    print("A\n", A)
    print("B\n", B)

    # A 역행렬 구해서
    Ainv = np.linalg.inv(A)
    print("Ainv\n", Ainv)

    # x = A^-1 * B
    x = np.einsum("ij,jk->ik", Ainv, B)
    print("x\n", x)