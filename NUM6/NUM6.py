import math
import random
import matplotlib.pyplot as plt
from scipy.linalg import qr, eig

# Definiowanie macierzy M
M = [
    [9, 2, 0, 0],
    [2, 4, 1, 0],
    [0, 1, 3, 1],
    [0, 0, 1, 2]
]


# Funkcja do mnożenia macierzy
def multiply_matrices(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Liczba kolumn macierzy A musi być równa liczbie wierszy macierzy B.")

    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result


# Funkcja do mnożenia macierzy przez wektor
def multiply_matrix_vector(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


# Funkcja do normalizacji wektora
def normalize_vector(v):
    norm = math.sqrt(sum(x ** 2 for x in v))
    return [x / norm for x in v]


# Funkcja do obliczania wartości własnej metodą potęgową
def power_method(A, tol=1e-10, max_iter=1000):
    n = len(A)
    v = [random.random() for _ in range(n)]
    v = normalize_vector(v)
    lambda_old = 0
    log_errors = []

    for _ in range(max_iter):
        w = multiply_matrix_vector(A, v)
        v = normalize_vector(w)
        lambda_new = sum(v[i] * multiply_matrix_vector(A, v)[i] for i in range(n))

        error = abs(lambda_new - lambda_old)
        log_errors.append(math.log10(error) if error > 0 else -float('inf'))

        if error < tol:
            break
        lambda_old = lambda_new

    return lambda_new, v, log_errors


# Funkcja do algorytmu QR (prosta implementacja)
def qr_algorithm(A, max_iter=1000, tol=1e-10):
    def transpose_matrix(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    def gram_schmidt(M):
        rows, cols = len(M), len(M[0])
        Q = [[0] * cols for _ in range(rows)]
        for j in range(cols):
            v = [M[i][j] for i in range(rows)]
            for k in range(j):
                r = sum(Q[i][k] * v[i] for i in range(rows))
                for i in range(rows):
                    v[i] -= r * Q[i][k]
            norm = math.sqrt(sum(x ** 2 for x in v))
            for i in range(rows):
                Q[i][j] = v[i] / norm
        R = [[sum(Q[i][k] * M[i][j] for i in range(rows)) for j in range(cols)] for k in range(cols)]
        return Q, R

    def matrix_product(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(len(A[0]))) for j in range(len(B[0]))] for i in range(len(A))]

    n = len(A)
    A_k = [row[:] for row in A]  # Deep copy
    diag_elements = []
    for _ in range(max_iter):
        Q, R = gram_schmidt(A_k)
        A_k = matrix_product(R, Q)
        diag_elements.append([A_k[i][i] for i in range(n)])

        off_diag = sum(A_k[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if math.sqrt(off_diag) < tol:
            break

    return A_k, diag_elements


# (a) Metoda potęgowa
lambda_max, eigenvector, log_errors = power_method(M)

print("Metoda potęgowa:")
print("Największa wartość własna:", lambda_max)
print("Odpowiadający wektor własny:", eigenvector)

# Wykres zbieżności metody potęgowej
plt.figure(figsize=(8, 6))
plt.plot(range(len(log_errors)), log_errors, marker='o')
plt.title('Zbieżność metody potęgowej (w skali logarytmicznej)')
plt.xlabel('Liczba iteracji')
plt.ylabel('Logarytm błędu')
plt.grid()
plt.show()

# (b) Algorytm QR
A_final, diag_evolution = qr_algorithm(M)

print("\nAlgorytm QR:")
print("Wartości własne:", [A_final[i][i] for i in range(len(A_final))])

# Wykres zbieżności dla algorytmu QR (w skali logarytmicznej)
log_errors_qr = []

# Obliczanie logarytmicznych błędów dla algorytmu QR
for diag in diag_evolution:
    current_error = sum(abs(A_final[i][i] - diag[i]) for i in range(len(diag)))
    log_errors_qr.append(math.log10(current_error) if current_error > 0 else -float('inf'))

# Wykres zbieżności algorytmu QR
plt.figure(figsize=(8, 6))
plt.plot(range(len(log_errors_qr)), log_errors_qr, marker='o')
plt.title('Zbieżność algorytmu QR (w skali logarytmicznej)')
plt.xlabel('Liczba iteracji')
plt.ylabel('Logarytm błędu')
plt.grid()
plt.show()

# Wykres elementów diagonalnych w funkcji iteracji
plt.figure(figsize=(8, 6))
for i in range(len(diag_evolution[0])):
    plt.plot(range(len(diag_evolution)), [row[i] for row in diag_evolution], label=f'Diagonal {i + 1}')
plt.title('Ewolucja elementów diagonalnych w algorytmie QR')
plt.xlabel('Liczba iteracji')
plt.ylabel('Wartość elementu diagonalnego')
plt.legend()
plt.grid()
plt.show()

# (c) Przedstawienie wyników przy pomocy biblioteki numerycznej
# Porównanie wartości własnych z metod wbudowanych
eigvals, eigvecs = eig(M)
# Zmiana formatu na 16 miejsc po przecinku
eigvals_formatted = [f"{val.real:.15f}" for val in eigvals]
# Usunięcie cudzysłowów w wyświetlaniu wartości
eigvals_formatted_no_quotes = ", ".join(eigvals_formatted)

# Wyniki z funkcji wbudowanej
print("\nPorownanie wynikow z funkcją wbudowaną (scipy.linalg.eig):")
print("Wartości własne: [", eigvals_formatted_no_quotes, "]")