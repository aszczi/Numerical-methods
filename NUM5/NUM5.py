import numpy as np
import matplotlib.pyplot as plt
import time

def create_matrix(N, d):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = d
        if i > 0:
            A[i, i - 1] = 0.5
        if i < N - 1:
            A[i, i + 1] = 0.5
        if i > 1:
            A[i, i - 2] = 0.1
        if i < N - 2:
            A[i, i + 2] = 0.1
    return A


def create_vector_b(N):
    return np.arange(1, N + 1)

def calculate_error(x1, x2):
    return np.max(np.abs(x1 - x2))


def print_first_and_last_elements(vec, name):
    print(f"{name}: [{', '.join(map(str, vec[:5]))} ... {', '.join(map(str, vec[-5:]))}]")


def create_initial_guess(N, value):
    return np.full(N, value)


def jacobi_method(A, b, exact_solution, initial_guess, max_iter=1000, tol=1e-10):
    N = len(b)
    x = initial_guess.copy()
    x_new = np.zeros_like(x)
    errors = []

    for _ in range(max_iter):
        for i in range(N):
            sum_ = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = sum_ / A[i, i]

        errors.append(calculate_error(x_new, exact_solution))
        if calculate_error(x_new, x) <= tol:
            break
        x = x_new.copy()

    return x, errors


def gauss_seidel_method(A, b, exact_solution, initial_guess, max_iter=1000, tol=1e-10):
    N = len(b)
    x = initial_guess.copy()
    errors = []

    for _ in range(max_iter):
        for i in range(N):
            sum_ = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = sum_ / A[i, i]

        errors.append(calculate_error(x, exact_solution))
        if errors[-1] <= tol:
            break

    return x, errors


def plot_errors(jacobi_errors, gauss_errors, d, init_val):
    plt.figure(figsize=(8, 6))
    iterations = np.arange(max(len(jacobi_errors), len(gauss_errors)))

    if jacobi_errors:
        plt.plot(iterations[:len(jacobi_errors)], np.log10(jacobi_errors), label="Jacobi", linewidth=2)
    if gauss_errors:
        plt.plot(iterations[:len(gauss_errors)], np.log10(gauss_errors), label="Gauss-Seidel", linewidth=2)

    plt.title(f"Błąd w kolejnych iteracjach (d = {d}, Start = {init_val})")
    plt.xlabel("Iteracja")
    plt.ylabel("log10(Błąd)")
    plt.grid()
    plt.legend()
    plt.show()


def main():
    N = 200
    d_values = [-1.3, 0.1, 0.5, 0.9, 1.1, 1.5, 4.0, 7.0]
    initial_values = [-1.0, 0.0, 10.0, 100.0]
    max_iter = 500
    tol = 1e-6

    for d in d_values:
        print(f"\n=== Testowanie dla d = {d} ===")
        A = create_matrix(N, d)
        b = create_vector_b(N)
        exact_solution = np.linalg.solve(A, b)

        print_first_and_last_elements(exact_solution, "Eigen LU")

        for init_val in initial_values:
            initial_guess = create_initial_guess(N, init_val)
            print(f"\nPunkt startowy: {init_val}")

            start_time = time.time()
            jacobi_solution, jacobi_errors = jacobi_method(A, b, exact_solution, initial_guess, max_iter, tol)
            jacobi_time = time.time() - start_time

            start_time = time.time()
            gauss_solution, gauss_errors = gauss_seidel_method(A, b, exact_solution, initial_guess, max_iter, tol)
            gauss_time = time.time() - start_time

            plot_errors(jacobi_errors, gauss_errors, d, init_val)

            print("Metoda Jacobiego:")
            print_first_and_last_elements(jacobi_solution, "Rozwiązanie Jacobi")
            print(f"  Czas wykonania: {jacobi_time:.6f} s")
            print(f"  Błąd końcowy: {jacobi_errors[-1] if jacobi_errors else 'Brak'}")
            print(f"  Liczba iteracji: {len(jacobi_errors)}")

            print("Metoda Gaussa-Seidela:")
            print_first_and_last_elements(gauss_solution, "Rozwiązanie Gauss-Seidel")
            print(f"  Czas wykonania: {gauss_time:.6f} s")
            print(f"  Błąd końcowy: {gauss_errors[-1] if gauss_errors else 'Brak'}")
            print(f"  Liczba iteracji: {len(gauss_errors)}")


if __name__ == "__main__":
    main()