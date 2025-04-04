import time
import matplotlib.pyplot as plt #do narysowania wykresu
from functools import reduce

# Funkcja do pomiaru czasu wykonania dla danego n
def measure_time(n):
    # Uzupełnienie diagonali macierzy A 
    matrix = []
    matrix.append([0] + [0.3] * (n - 1))
    matrix.append([1.01] * n)
    matrix.append([0.2 / i for i in range(1, n)] + [0])
    matrix.append([0.15 / i**3 for i in range(1, n-1)] + [0] + [0])

    # Stworzenie wektora wyrazów wolnych
    x = list(range(1, n + 1))

    # Start pomiaru czasu
    start = time.time()

    # Rozkład LU
    for i in range(1, n-2):
        matrix[0][i] = matrix[0][i] / matrix[1][i - 1]
        matrix[1][i] = matrix[1][i] - matrix[0][i] * matrix[2][i - 1]
        matrix[2][i] = matrix[2][i] - matrix[0][i] * matrix[3][i - 1]

    matrix[0][n-2] = matrix[0][n-2] / matrix[1][n-3]
    matrix[1][n-2] = matrix[1][n-2] - matrix[0][n-2] * matrix[2][n-3]
    matrix[2][n-2] = matrix[2][n-2] - matrix[0][n-2] * matrix[3][n-3]

    matrix[0][n-1] = matrix[0][n-1] / matrix[1][n-2]
    matrix[1][n-1] = matrix[1][n-1] - matrix[0][n-1] * matrix[2][n-2]

    # Podstawianie w przód
    for i in range(1, n):
        x[i] = x[i] - matrix[0][i] * x[i - 1]

    # Podstawiania w tył
    x[n-1] = x[n-1] / matrix[1][n-1]
    x[n-2] = (x[n-2] - matrix[2][n-2] * x[n-1]) / matrix[1][n-2]

    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - matrix[3][i] * x[i + 2] - matrix[2][i] * x[i + 1]) / matrix[1][i]

    # Obliczanie wartości wyznacznika macierzy
    wyznacznik = reduce(lambda a, b: a * b, matrix[1])

    # Koniec pomiaru czasu
    end = time.time() - start

    return end, x, wyznacznik

# Inicjalizacja n_values w pętli
n_values = [100 + 100 * i for i in range(300)]
times = []

# Pomiar czasu dla wartości n z zakresu do 100 000
for n in n_values:
    execution_time, _, _ = measure_time(n)
    times.append(execution_time)

# Wykres
plt.plot(n_values, times, marker='o')
plt.xlabel('n')
plt.ylabel('Czas wykonania (s)')
plt.title('Zależność czasu wykonania od parametru n')
plt.grid(True)
plt.show()

# Wyświetlenie x i wyznacznika dla n = 300
n = 300
_, x, wyznacznik = measure_time(n)
print(f"Rozwiązanie x dla n={n}: {x}")
print(f"Wyznacznik macierzy A dla n={n}: {wyznacznik}")