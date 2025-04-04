import numpy as np                  # do sprawdzenia wynikow
import matplotlib.pyplot as plt     # do wygenerowania wykresu
import time                         #do mierzenia czasu wykonania programu


def sherman():
    M = []
    M.append([4]*n)
    M.append([2]*(n-1) + [0])

    start = time.time()

    # Backward subtitution dla obu równań
    z = [0]*n
    x = [0]*n
    z[n-1] = b[n-1] / M[0][n-1]
    x[n-1] = 1 / M[0][n-1]

    for i in range(n - 2, -1, -1):
        z[i] = (b[n-2] - M[1][i] * z[i+1]) / M[0][i]
        x[i] = (1 - M[1][i] * x[i+1]) / M[0][i]


    delta = sum(z)/(1+sum(x))

    # Wyliczenie wyniku
    y=[]
    for i in range(len(z)):
        y.append(z[i]-x[i]*delta)

    end = time.time()-start

    if(not test):
        print(y)
    else:
        return end
        
def from_numpy():
    A = np.ones((n, n))
    A += np.diag([4] * n)
    A += np.diag([2] * (n - 1), 1)

    start = time.time()
    np.linalg.solve(A, b)
    return time.time()-start
    
n = 120
test = False
b = [5]*n
sherman()



# Testowanie czasu działania numpy.linalg.solve
#N = []  # Lista wartości N
#times = []  # Lista czasów działania

#for i in range(50, 3000, 100):  # Testujemy dla N od 50 do 2000 z krokiem 100
#    n = i
#    b = [5] * n

    # Pomiar czasu dla numpy.linalg.solve
#    elapsed_time = from_numpy() * 1_000_000  # Konwersja na mikrosekundy
#    N.append(n)
#    times.append(elapsed_time)

# Tworzenie wykresu dla numpy.linalg.solve
#plt.figure(figsize=(10, 6))
#plt.plot(N, times, marker='o', label='Czas działania numpy.linalg.solve', color='orange')
#plt.grid(True)
#plt.yscale('log')
#plt.title('Czas działania numpy.linalg.solve w zależności od N')
#plt.xlabel('Rozmiar N (długość wektora b i macierzy A)')
#plt.ylabel('Czas (mikrosekundy)')
#plt.legend()
#plt.show()


# Testowanie czasu działania funkcji sherman
#test = True
#N = []  # Lista wartości N
#times = []  # Lista czasów działania

#for i in range(50, 3000, 100):  # Testujemy dla N od 50 do 2000 z krokiem 100
#    n = i
#    b = [5] * n

    # Pomiar czasu dla funkcji sherman
#    elapsed_time = sherman() * 1_000_000  # Konwersja na mikrosekundy
#    N.append(n)
#    times.append(elapsed_time)

# Tworzenie wykresu
#plt.figure(figsize=(10, 6))
#plt.plot(N, times, marker='o', label='Czas działania funkcji sherman')
#plt.grid(True)
#plt.title('Czas działania funkcji sherman w zależności od N')
#plt.xlabel('Rozmiar N (długość wektora b i macierzy A)')
#plt.ylabel('Czas (mikrosekundy)')
#plt.legend()
#plt.show()