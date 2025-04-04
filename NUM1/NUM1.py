import numpy as np
import matplotlib.pyplot as plt

# Funkcja f(x) = sin(x^3)
def f(x):
    return np.sin(x**3)

# Pochodna analityczna f'(x) = 3*x^2 * cos(x^3)
def df(x):
    return 3 * x**2 * np.cos(x**3)

# Przybliżenie pochodnej ze wzoru (a) Dhf(x) = (f(x+h) - f(x)) / h
def Dhf_a(f, x, h):
    return (f(x + h) - f(x)) / h

# Przybliżenie pochodnej ze wzoru (b) Dhf(x) = (f(x+h) - f(x-h)) / (2h)
def Dhf_b(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Funkcja do analizy błędu dla różnych typów zmiennoprzecinkowych
def analyze_error(x, h_values, float_type):
    f_float = lambda x: f(np.array(x, dtype=float_type))
    df_float = lambda x: df(np.array(x, dtype=float_type))
    
    # Rzeczywista pochodna w punkcie x
    real_df = df_float(x)
    
    # Obliczanie przybliżeń Dhf_a i Dhf_b
    errors_a = []
    errors_b = []
    
    for h in h_values:
        approx_a = Dhf_a(f_float, x, h)
        approx_b = Dhf_b(f_float, x, h)
        errors_a.append(np.abs(approx_a - real_df))
        errors_b.append(np.abs(approx_b - real_df))
    
    return np.array(errors_a), np.array(errors_b)
    
x = 0.2
h_values = np.logspace(-16, 0, 300)  # Wartości h od bardzo małych do większych w skali logarytmicznej

###################################
#Porównanie wyników z float 32
# Parametry

# Analiza błędu dla typu float32
errors_a_float32, errors_b_float32 = analyze_error(x, h_values, np.float32)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_a_float32, label='Wzór (a), float32', color='green')
plt.loglog(h_values, errors_b_float32, label='Wzór (b), float32', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dhf(x) - f\'(x)| dla f(x) = sin(x^3), x = 0.2 dla float 32')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()

####################################
#Porównanie wyników z float 64
# Parametry

# Analiza błędu dla typu float64 (double)
errors_a_float64, errors_b_float64 = analyze_error(x, h_values, np.float64)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float64
plt.loglog(h_values, errors_a_float64, label='Wzór (a), float64', color='green')
plt.loglog(h_values, errors_b_float64, label='Wzór (b), float64', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dhf(x) - f\'(x)| dla f(x) = sin(x^3), x = 0.2 dla float 64')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()


####################################
#porównanie wyników dla wykresu a
# Parametry

# Analiza błędu dla typu float32
errors_a_float32, errors_b_float32 = analyze_error(x, h_values, np.float32)

# Analiza błędu dla typu float64 (double)
errors_a_float64, errors_b_float64 = analyze_error(x, h_values, np.float64)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_a_float32, label='Wzór (a), float32', color='green')

# Wykres dla float64
plt.loglog(h_values, errors_a_float64, label='Wzór (a), float64', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dhf(x) - f\'(x)| dla f(x) = sin(x^3), x = 0.2 dla wzoru a')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()


####################################
#Porownanie danych dla wykresu b
# Parametry

# Analiza błędu dla typu float32
errors_a_float32, errors_b_float32 = analyze_error(x, h_values, np.float32)

# Analiza błędu dla typu float64 (double)
errors_a_float64, errors_b_float64 = analyze_error(x, h_values, np.float64)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_b_float32, label='Wzór (b), float32', color='green')

# Wykres dla float64
plt.loglog(h_values, errors_b_float64, label='Wzór (b), float64', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dhf(x) - f\'(x)| dla f(x) = sin(x^3), x = 0.2 dla wzoru b')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()

###################################################
#Wykresy dla innych przykładowych funkcji:

# Funkcja g(x) = x^4
def g(x):
    return x**4

# Pochodna analityczna g'(x) = 4*x^3
def dg(x):
    return 4 * x**3

# Przybliżenie pochodnej ze wzoru (a) Dhg(x) = (g(x+h) - g(x)) / h
def Dhg_a(g, x, h):
    return (g(x + h) - g(x)) / h

# Przybliżenie pochodnej ze wzoru (b) Dhg(x) = (g(x+h) - g(x-h)) / (2h)
def Dhg_b(g, x, h):
    return (g(x + h) - g(x - h)) / (2 * h)

# Funkcja do analizy błędu dla różnych typów zmiennoprzecinkowych
def analyze_error(x, h_values, float_type):
    g_float = lambda x: g(np.array(x, dtype=float_type))
    dg_float = lambda x: dg(np.array(x, dtype=float_type))
    
    # Rzeczywista pochodna w punkcie x
    real_dg = dg_float(x)
    
    # Obliczanie przybliżeń Dhf_a i Dhf_b
    errors_a = []
    errors_b = []
    
    for h in h_values:
        approx_a = Dhg_a(g_float, x, h)
        approx_b = Dhg_b(g_float, x, h)
        errors_a.append(np.abs(approx_a - real_dg))
        errors_b.append(np.abs(approx_b - real_dg))
    
    return np.array(errors_a), np.array(errors_b)

###################################
#Porównanie wyników z float 32
# Parametry
x = 1  # Analizujemy punkt x = 1

# Analiza błędu dla typu float32
errors_a_float32, errors_b_float32 = analyze_error(x, h_values, np.float32)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_a_float32, label='Wzór (a), float32', color='green')
plt.loglog(h_values, errors_b_float32, label='Wzór (b), float32', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dhg(x) - g\'(x)| dla g(x) = x^4, x = 1 dla float 32')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()
######################################
#Wykres dla g(x) = x^4  porównanie wyników z Float 64
x = 1  # Analizujemy punkt x = 1

# Analiza błędu dla typu float64
errors_a_float64, errors_b_float64 = analyze_error(x, h_values, np.float64)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_a_float64, label='Wzór (a), float64', color='green')
plt.loglog(h_values, errors_b_float64, label='Wzór (b), float64', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dhg(x) - g\'(x)| dla g(x) = x^4, x = 1 dla float 64')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()

######################################
#Wykres dla h(x)=cos(x^2) w punkcie x = 0.5

# Funkcja h(x) = cos(x^2)
def h(x):
    return np.cos(x**2)

# Pochodna analityczna h'(x) = -2x*sin(x^2)
def dh(x):
    return -2 * x * np.sin(x**2)

# Przybliżenie pochodnej ze wzoru (a) Dh(x) = (h(x+h) - h(x)) / h
def Dh_a(h, x, h_step):
    return (h(x + h_step) - h(x)) / h_step

# Przybliżenie pochodnej ze wzoru (b) Dh(x) = (h(x+h) - h(x-h)) / (2h)
def Dh_b(h, x, h_step):
    return (h(x + h_step) - h(x - h_step)) / (2 * h_step)

# Funkcja do analizy błędu dla różnych typów zmiennoprzecinkowych
def analyze_error(x, h_values, float_type):
    h_float = lambda x: h(np.array(x, dtype=float_type))
    dh_float = lambda x: dh(np.array(x, dtype=float_type))
    
    # Rzeczywista pochodna w punkcie x
    real_dh = dh_float(x)
    
    # Obliczanie przybliżeń Dh_a i Dh_b
    errors_a = []
    errors_b = []
    
    for h_step in h_values:
        approx_a = Dh_a(h_float, x, h_step)
        approx_b = Dh_b(h_float, x, h_step)
        errors_a.append(np.abs(approx_a - real_dh))
        errors_b.append(np.abs(approx_b - real_dh))
    
    return np.array(errors_a), np.array(errors_b)

###################################
#Porównanie wyników z float 64
# Parametry
h_values = np.logspace(-11, 0, 400)  # Wartości h od bardzo małych do większych w skali logarytmicznej

x = 0.5  # Analizujemy punkt x = 0.5

# Analiza błędu dla typu float64
errors_a_float64, errors_b_float64 = analyze_error(x, h_values, np.float64)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_a_float64, label='Wzór (a), float64', color='green')
plt.loglog(h_values, errors_b_float64, label='Wzór (b), float64', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dh(x) - h\'(x)| dla h(x) = cos(x^2), x = 0.5 dla float 64')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()

#######################################
#Tensamże wykres dla x = 0.0001

h_values = np.logspace(-11, 0, 400)  # Wartości h od bardzo małych do większych w skali logarytmicznej

x = 0.01  # Analizujemy punkt x = 0.01

# Analiza błędu dla typu float64
errors_a_float64, errors_b_float64 = analyze_error(x, h_values, np.float64)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

# Wykres dla float32
plt.loglog(h_values, errors_a_float64, label='Wzór (a), float64', color='green')
plt.loglog(h_values, errors_b_float64, label='Wzór (b), float64', color='blue')

# Dodatkowe informacje
plt.title('Błąd |Dh(x) - h\'(x)| dla h(x) = cos(x^2), x = 0.01 dla float 64')
plt.xlabel('h')
plt.ylabel('Błąd')
plt.legend()
plt.grid(True, which="both", ls="--")

# Wyświetlenie wykresu
plt.show()