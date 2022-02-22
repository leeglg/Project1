import numpy as np
# dot 함수. 인자 2개 받는걸로 사용.
# 인자 2개 종류에 따라 연산이 다름.

# scalar 두개 받으면 곱연산  =>  * 쓰는게 더 편하다
print(np.dot(5, 7))

# 같은 크기 벡터 두개면 벡터 내적
print(np.dot([1, 2],[3, 4]))
# 다른 크기 벡터 면?
# print(np.dot([1,2,3],[1,2]))
# shape 안 맞는다고 오류

# 같은 크기 행렬 두개면 행렬 곱  =>  @ 쓰는게 더 편하다
a = np.array([[1, 0], [0, 1]])
b = np.array([[4, 1], [2, 2]])
print(np.dot(a, b))

print("\n")
#  a가 N차원 행렬이고, b가 1차원 행렬이라면, a의 마지막 축에 b를 곱하여 더한 값    쓸 일 없을듯.
c = np.arange(2*2*2).reshape(2, 2, 2)
d = np.array([1,1])
print("c\n", c)
print("\nd")
print(np.dot(c, d))




print("\n\n\n\n\neinsum")
# einsum 함수. 행렬 연산 표기 편하게 하기 위한 함수. Einstein Summation Convention 따름.

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("A\n", A)

#전치
T = np.einsum("ij->ji", A)
print("T\n", T)

#합
sum = np.einsum('ij->', A)
print("sum\n", sum)
csum = np.einsum('ij->j', A)
print("csum\n", csum)
rsum = np.einsum('ij->i', A)
print("rsum\n", rsum)

B = np.array([[-1, 0, 1]])
#행렬곱
mul = np.einsum("ij,kj->ik", A, B)  # 텐서의 각 차원들 돌 변수를 순서대로 입력 -> 남길 차원들
print("mul\n", mul)

#내적 dot product  결과 스칼라 하나
dot = np.einsum("ij,ij->", A, A)
print("dot\n", dot)

temp1 = np.array([[1, 2, 3]]) # 행렬로 된거임 1 * 3 ]]
temp2 = np.array([[4, 5, 6]])
#외적 outer product  벡터간 행렬곱
outer = np.einsum("i,j->ij", temp1[0], temp2[0])
print("outer\n", outer)

#하다마르 곱
ha = np.einsum("ij,ij->ij", A, A)
print("ha\n", ha)

#3차원 이상의 텐서에 대한 계산.   2차원 행렬곱의 반복 연산
te = np.random.randint(0, 100, size=(3, 2, 5))
te2 = np.random.randint(0, 100, size=(3, 5, 3))
bmm = np.einsum("ijk,ikl->ijl", te, te2)
print("bmm\n", bmm)  # 3,2,3 크기의 텐서  i 유지, jk,kl->jl  2*5,5*3->2*3

# matrix diagonal  대각 성분만 뱉기
te3 = np.random.randint(0, 10, size=(3, 3))
diag = np.einsum("ii->i", te3)
print("te3\n", te3)
print("diag\n", diag)

# matrix trace 대각 성분 합
trace = np.einsum("ii->", te3)
print("trace\n", trace)