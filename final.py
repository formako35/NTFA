import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


INITIAL_SURFACE_TEMP = 30
INITIAL_INNER_TEMP = 3

def solve_temperature(L,T,N,M,A,B,C,D,):
    dr = L / (N - 1)
    dt = T / M
    x = np.linspace(0, L, N)
    t = np.linspace(0, T, M)

    r1 = B * dt / (A * dr ** 2)
    r2 = C * dt / (A * dr ** 2)
    r3 = dt / A

    def Q(t):
        return 20

    main_diag = np.ones(N) + 2 * r1
    off_diag = -1 * r1 * np.ones(N-1)
    A_matrix = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).toarray()

    A_matrix[0, 0] = A_matrix[N-1, N-1] = 1
    A_matrix[0, 1] = A_matrix[N-1, N-2] = 0

    u = np.zeros((M, N))
    u[0, :] = INITIAL_INNER_TEMP

    for n in range(0, M - 1):
        T_previous = u[n, :]

        sqrt_T_vector = (1/np.sqrt(T_previous))
        main_diag = -1*np.ones(N)
        off_diag = 1*np.ones(N - 1)
        Omega2 = diags([main_diag, off_diag], [ 0, 1]).toarray()
        Omega2[N-1, N-1] = Omega2[N-1, N-2] = 0
        Omega2 = r2 * Omega2

        b = T_previous + r3 * (-D + Q(t[n + 1]))
        b[0] = INITIAL_SURFACE_TEMP
        b[-1] = INITIAL_INNER_TEMP

        u_next = spsolve(A_matrix, b-Omega2@sqrt_T_vector)

        u[n + 1, :] = u_next

    plt.figure(figsize=(10, 8))
    plt.imshow(u.T, aspect='auto', extent=[0, T, 0, L], cmap='hot')
    plt.colorbar(label='Temperature')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Heat Equation Solution with Source Term\n'
              f'L={L} T={T} N={N} M={M}\n'
              f'A={A} B={B} C={C} D={D}')

def main():
    solve_temperature(L=1, T=0.1, N=200, M=100, A=1, B=1, C=0, D=0)
    solve_temperature(L=1, T=0.1, N=200, M=100, A=1, B=1, C=2, D=0)
    plt.show()

if __name__ == '__main__':
    main()
