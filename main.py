import copy
import numpy as np


def get_LU(matrix):
    A = copy.deepcopy(matrix)
    n = len(A[0])
    rows = []
    L = np.eye(n)
    for k in range(n-1):
        r = np.argmax(np.abs(A[k:, k])) + k
        print(A[k:, k])
        print(f"r: {r}")
        if A[k, r] == 0:
            print("The maximum pivot is zero. The system is not solvable!")
            return None
        rows.append(r)
        A[[k, r]] = A[[r, k]]
        print(A)
        print(20*"@@")
        for i in range(k+1, n):
            A[i, k] = -A[i, k]/A[k, k]
            L[i, k] = A[i, k]
            A[i, k+1:] = A[i, k+1:] + A[k, k+1:] * A[i, k]
            print(A)
            print(20*"@@")
    # Create Ms
    m_list = []
    for i in range(n-1):
        M = np.eye(n)
        M[i+1:, i] = A[i+1:, i]
        m_list.append(M)
        print("M: ")
        print(M)
        print(20*'##')
    # Create Ps
    p_list = []
    P = np.eye(n)
    for i, row in enumerate(rows):
        p = np.eye(n)
        p[[i, row]] = p[[row, i]]
        p_list.append(p)
        P = np.dot(p, P)
        print("P: ")
        print(p)
        print(20*'##')
    print("Final P: ")
    print(P)
    print(20*"@@")
    # Create M
    M = np.dot(m_list[0], p_list[0])
    for i in range(1, len(p_list)):
        MP = np.dot(m_list[i], p_list[i])
        M = np.dot(MP, M)
    print("final M: ")
    print(M)
    print(20*"==")
    U = np.triu(A, k=0)
    L = np.dot(P, np.linalg.inv(M))
    return L, U, P


def solve_forward(L, b):
    n = len(b)
    y = np.zeros(n, dtype="float32")
    y[0] = b[0]/L[1, 1]
    for i in range(1, n):
        y[i] = b[i]
        for j in range(1, i):
            y[i] = y[i] - L[i, j]*y[j]
        y[i] = y[i] / L[i, i]
    return y


def solve_backward(U, b):
    n = len(b)
    x = np.zeros(n, dtype="float32")
    x[n-1] = b[n-1]/U[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] = x[i] - U[i, j]*x[j]
        x[i] = x[i] / U[i, i]
    return x


def solve_equation(A, b, verbose=True):
    b = np.array(b, dtype="float32")
    A = np.array(A, dtype="float32")
    try:
        L, U, P = get_LU(A)
    except Exception as e:
        print(f"An Error Accured: {e}")
        return None
    b = np.dot(P, b)
    y = solve_forward(L, b)
    x = solve_backward(U, y)
    if verbose:
        print("A: ")
        print(A)
        print(20*"--")
        print("b: ")
        print(b)
        print(20*"--")
        print("L: ")
        print(L)
        print(20*"--")
        print("U:")
        print(U)
        print(20*"--")
        print("LU:")
        print(np.dot(L, U))
        print(f"x: {x}")
    return x


if __name__ == "__main__":
    A = [[2, 6, -2],
         [-4, 4, 2],
         [1, 2, 3]]
    b = [6, 2, 6]
    # A = [[1, 1, 1], [4, 3, -1], [3, 5, 3]]
    # b = [1,6,4]
    # print(np.array(A, dtype="float32"))
    solve_equation(A, b)
    # print(solve_backward(np.array([[2,6,-2], [0,-2,4], [0,0,2]]), np.array([6,2,2])))