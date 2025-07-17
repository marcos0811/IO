import pandas as pd                  # Para leer archivos Excel y manejar datos estructurados
import random                        # Para hacer elecciones aleatorias
import math                          # Para funciones matemáticas como exp()
import matplotlib.pyplot as plt      # Para generar gráficas
from collections import defaultdict  # Para crear diccionarios con listas por defecto

# === CARGA DE DATOS DESDE EL ARCHIVO EXCEL ===
archivo_excel = r"C:\Users\eddis\Downloads\TVtronics-datos.xlsx"
datos_excel = pd.read_excel(archivo_excel, sheet_name=None)  # Carga todas las hojas

# Se seleccionan las hojas relevantes del archivo
productos = datos_excel["Datos de productos"]
setup_df = datos_excel["Tiempos de cambio"].set_index("Unnamed: 0")

# Diccionarios que guardan la información por modelo
modelos = productos["Modelo"].tolist()
demanda = dict(zip(modelos, productos["Demanda"]))  # Unidades a producir
costo_unitario = dict(zip(modelos, productos["Costo unitario (USD)"]))  # Costo por unidad
volumen = dict(zip(modelos, productos["Volumen (m³/unidad)"]))  # Volumen ocupado en bodega

# Diccionario que guarda los tiempos de procesamiento por línea y modelo, en minutos
procesamiento = {
    modelo: [
        productos.loc[i, "Tiempo L1 (h)"] * 60,
        productos.loc[i, "Tiempo L2 (h)"] * 60,
        productos.loc[i, "Tiempo L3 (h)"] * 60
    ]
    for i, modelo in enumerate(modelos)
}

# Diccionario con los tiempos de cambio de modelo a modelo (setup)
setup = setup_df.to_dict()

# === PARÁMETROS DEL PROBLEMA ===
LINEAS = [0, 1, 2]             # Representa las 3 líneas de producción
TIEMPO_MAX = 2880             # Máximo tiempo disponible por línea (en minutos)
BODEGA_MAX = 320              # Máxima capacidad de bodega (en metros cúbicos)
COSTO_SETUP = 3               # Costo por minuto de tiempo de cambio entre modelos

# === GENERAR UNA SOLUCIÓN ALEATORIA INICIAL ===
def generar_solucion():
    # Asigna cada modelo a una línea aleatoria
    return [(modelo, random.choice(LINEAS)) for modelo in modelos]

# === FUNCIÓN PARA EVALUAR UNA SOLUCIÓN ===
def evaluar(sol):
    tiempo_lineas = {l: 0 for l in LINEAS}  # Tiempo usado en cada línea
    volumen_total = 0
    costo = 0
    detalle = defaultdict(list)     # Detalle por línea: modelo, tiempo, volumen, costo
    secuencia = defaultdict(list)   # Orden de modelos por línea
    penalizacion = 0

    # Recorremos cada modelo asignado a una línea
    for modelo, linea in sol:
        d = demanda[modelo]
        t = procesamiento[modelo][linea] * d  # Tiempo total
        v = volumen[modelo] * d               # Volumen total
        c = costo_unitario[modelo] * d        # Costo de producción

        tiempo_lineas[linea] += t
        volumen_total += v
        costo += c
        detalle[linea].append((modelo, t, v, c))
        secuencia[linea].append(modelo)

    # Penalización por exceder tiempo por línea
    for l in LINEAS:
        if tiempo_lineas[l] > TIEMPO_MAX:
            penalizacion += (tiempo_lineas[l] - TIEMPO_MAX) * 10

    # Penalización por exceder el volumen total permitido
    if volumen_total > BODEGA_MAX:
        penalizacion += (volumen_total - BODEGA_MAX) * 20

    # Costo de setup entre modelos en una misma línea
    for l in LINEAS:
        modelos_linea = secuencia[l]
        for i in range(1, len(modelos_linea)):
            ant = modelos_linea[i - 1]
            act = modelos_linea[i]
            costo += setup[ant][act] * COSTO_SETUP

    # Retorna todos los datos necesarios para analizar la solución
    return {
        "costo_total": costo,
        "penalizacion": penalizacion,
        "costo_final": costo + penalizacion,
        "tiempos": tiempo_lineas,
        "volumen": volumen_total,
        "detalle": detalle,
        "orden": secuencia
    }

# === GENERAR UNA SOLUCIÓN VECINA CAMBIANDO UNA LÍNEA ===
def generar_vecino(sol):
    vecino = sol.copy()
    i = random.randint(0, len(vecino)-1)
    modelo, linea = vecino[i]
    nueva_linea = random.choice([l for l in LINEAS if l != linea])
    vecino[i] = (modelo, nueva_linea)
    return vecino

# === ALGORITMO DE RECOCIDO SIMULADO CLÁSICO ===
def recocido_simulado(sol_inicial, T_inicial=1000, T_final=1, alfa=0.95, max_iter=50):
    actual = sol_inicial
    eval_actual = evaluar(actual)
    mejor = actual
    eval_mejor = eval_actual
    historial = [eval_actual["costo_final"]]
    T = T_inicial  # Temperatura inicial

    # Mientras la temperatura no llegue al mínimo
    while T > T_final:
        for _ in range(max_iter):  # Iteraciones por nivel de temperatura
            vecino = generar_vecino(actual)
            eval_vecino = evaluar(vecino)
            delta = eval_vecino["costo_final"] - eval_actual["costo_final"]

            # Aceptar si mejora o si cumple probabilidad de aceptación
            if delta < 0 or random.random() < math.exp(-delta / T):
                actual = vecino
                eval_actual = eval_vecino
                if eval_vecino["costo_final"] < eval_mejor["costo_final"]:
                    mejor = vecino
                    eval_mejor = eval_vecino

            historial.append(eval_mejor["costo_final"])

        T *= alfa  # Se reduce la temperatura (enfriamiento)

    return eval_mejor, historial

# === GRAFICAR CONVERGENCIA DEL COSTO ===
def graficar_convergencia(hist):
    plt.plot(hist)
    plt.title("Convergencia del Recocido Simulado")
    plt.xlabel("Iteración")
    plt.ylabel("Costo Total")
    plt.grid()
    plt.show()

# === GRAFICAR TIEMPO POR LÍNEA ===
def graficar_tiempo(tiempos):
    plt.bar(["L1", "L2", "L3"], [tiempos[0], tiempos[1], tiempos[2]], color="salmon")
    plt.axhline(TIEMPO_MAX, color="green", linestyle="--", label="Tiempo máximo permitido")
    plt.title("Tiempo requerido por línea")
    plt.ylabel("Minutos")
    plt.legend()
    plt.show()

# === GRAFICAR VOLUMEN VS CAPACIDAD DE BODEGA ===
def graficar_volumen(vol):
    plt.bar(["Volumen total requerido"], [vol], color="skyblue")
    plt.axhline(BODEGA_MAX, color="green", linestyle="--", label="Espacio máximo en bodega")
    plt.ylabel("m³")
    plt.title("Volumen requerido vs capacidad de bodega")
    plt.legend()
    plt.show()

# === EJECUCIÓN PRINCIPAL DEL PROGRAMA ===
if __name__ == "__main__":
    sol_inicial = generar_solucion()
    resultado, historial = recocido_simulado(
        sol_inicial,
        T_inicial=1000,
        T_final=1,
        alfa=0.95,
        max_iter=50
    )

    print("\n========= PLAN FINAL =========")
    for l in LINEAS:
        print(f"\nLínea {l+1}:")
        for i, (modelo, t, v, c) in enumerate(resultado["detalle"][l], 1):
            print(f"  {i}. Modelo {modelo} | Tiempo: {round(t)} min | Volumen: {round(v, 2)} m³ | Costo: ${round(c)}")
        print(f"  Total: {round(resultado['tiempos'][l], 2)} / {TIEMPO_MAX} min")

    print(f"\nVolumen total: {round(resultado['volumen'], 2)} / {BODEGA_MAX} m³")
    print(f"Costo sin penalizaciones: ${round(resultado['costo_total'], 2)}")
    print(f"Penalización total: ${round(resultado['penalizacion'], 2)}")
    print(f"COSTO FINAL: ${round(resultado['costo_final'], 2)}")

    graficar_convergencia(historial)
    graficar_tiempo(resultado["tiempos"])
    graficar_volumen(resultado["volumen"])
