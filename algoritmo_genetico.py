import pandas as pd
import random
import time
import matplotlib.pyplot as plt

# ==========================
# PARÁMETROS DEL PROBLEMA
# ==========================

# Defino las constantes que puedo modificar fácilmente en caso de cambios en el problema
TIEMPO_DISPONIBLE = 48 * 60         # 48 horas por línea, convertidas a minutos
LINEAS = [0, 1, 2]                  # Las tres líneas de ensamblaje disponibles
ESPACIO_BODEGA = 320               # Capacidad total de almacenamiento en m³
COSTO_SETUP_MIN = 3                # Costo por minuto de setup
PENALIZACION_TIEMPO = 10           # Penalización por minuto extra usado
PENALIZACION_VOLUMEN = 20          # Penalización por m³ extra almacenado
NUM_GENERACIONES = 100             # Número de generaciones del algoritmo genético
TAM_POBLACION = 50                 # Tamaño de la población en cada generación

# ==========================
# PARTE 1: CARGA DE DATOS
# ==========================

# Cargo los datos del archivo Excel (ya debe estar en el mismo directorio)
excel_data = pd.read_excel("TVtronics-datos.xlsx", sheet_name=None)

# Divido las hojas en DataFrames separados
df_productos = excel_data["Datos de productos"]
df_setup = excel_data["Tiempos de cambio"].set_index("Unnamed: 0")

# Extraigo todos los modelos de televisores disponibles
modelos = df_productos["Modelo"].tolist()

# Mapeo los parámetros por modelo usando diccionarios para fácil acceso
demanda = dict(zip(modelos, df_productos["Demanda"]))
costo_unitario = dict(zip(modelos, df_productos["Costo unitario (USD)"]))
volumen = dict(zip(modelos, df_productos["Volumen (m³/unidad)"]))

# Paso los tiempos de producción por línea de horas a minutos
tiempo_procesamiento = {
    modelo: [
        df_productos.loc[i, "Tiempo L1 (h)"] * 60,
        df_productos.loc[i, "Tiempo L2 (h)"] * 60,
        df_productos.loc[i, "Tiempo L3 (h)"] * 60
    ]
    for i, modelo in enumerate(modelos)
}

# Guardo los tiempos de setup entre pares de modelos en un diccionario anidado
tiempo_setup = df_setup.to_dict()

# ==========================
# FUNCIÓN: VERIFICAR FACTIBILIDAD
# ==========================

def verificar_factibilidad():
    # Verifico si es posible satisfacer toda la demanda dentro de los límites
    tiempo_por_linea = [0, 0, 0]
    volumen_total = 0

    # Simulo producir toda la demanda en cada línea por separado
    for l in LINEAS:
        for modelo in modelos:
            unidades = demanda[modelo]
            tiempo_por_linea[l] += tiempo_procesamiento[modelo][l] * unidades
        print(f"Tiempo requerido si todo fuera en L{l+1}: {tiempo_por_linea[l]:.2f} min")

    # Sumo el volumen total requerido por todos los modelos
    for modelo in modelos:
        volumen_total += volumen[modelo] * demanda[modelo]

    # Muestro los resultados comparados con las capacidades máximas
    print(f"Volumen total requerido: {volumen_total:.2f} m³")
    print(f"Tiempo disponible por línea: {TIEMPO_DISPONIBLE} min")
    print(f"Tiempo total disponible (3 líneas): {TIEMPO_DISPONIBLE * len(LINEAS)} min")
    print(f"Espacio disponible en bodega: {ESPACIO_BODEGA} m³")

    # Verifico si alguna línea o el volumen excede su capacidad
    if any(t > TIEMPO_DISPONIBLE for t in tiempo_por_linea) or volumen_total > ESPACIO_BODEGA:
        print("El problema **NO** tiene una solución factible con las restricciones actuales.\n")
    else:
        print("El problema tiene una solución factible.\n")

    # Genero un gráfico para visualizar el uso de tiempo en cada línea y del espacio
    # Gráfico 1: Tiempo requerido por línea
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(["L1", "L2", "L3"], tiempo_por_linea, color='salmon')
    plt.axhline(y=TIEMPO_DISPONIBLE, color='green', linestyle='--', label='Tiempo máximo permitido')
    plt.title("Tiempo requerido por línea")
    plt.ylabel("Minutos")
    plt.legend()

    # Gráfico 2: Volumen requerido vs permitido
    plt.subplot(1, 2, 2)
    plt.bar(["Volumen total requerido"], [volumen_total], color='skyblue')
    plt.axhline(y=ESPACIO_BODEGA, color='green', linestyle='--', label='Espacio máximo en bodega')
    plt.title("Volumen requerido vs capacidad de bodega")
    plt.ylabel("m³")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==========================
# FUNCIÓN: ALGORITMO GENÉTICO
# ==========================

def crear_individuo():
    # Asigno aleatoriamente cada modelo a una línea (solución candidata)
    return [(modelo, random.choice(LINEAS)) for modelo in modelos]

def evaluar_individuo(individuo):
    # Evalúo cuánto cuesta una solución (producción + setup + penalizaciones)
    secuencia_por_linea = {l: [] for l in LINEAS}
    tiempo_usado = {l: 0 for l in LINEAS}
    asignacion_por_modelo = {m: [0, 0, 0] for m in modelos}
    volumen_total = 0
    costo_total = 0
    penalizacion = 0

    # Recorro cada modelo asignado a una línea
    for modelo, linea in individuo:
        unidades = demanda[modelo]
        asignacion_por_modelo[modelo][linea] += unidades
        secuencia_por_linea[linea].append(modelo)
        tiempo_usado[linea] += tiempo_procesamiento[modelo][linea] * unidades
        volumen_total += volumen[modelo] * unidades
        costo_total += costo_unitario[modelo] * unidades

    # Penalizo el exceso de tiempo en cualquier línea
    for l in LINEAS:
        if tiempo_usado[l] > TIEMPO_DISPONIBLE:
            penalizacion += (tiempo_usado[l] - TIEMPO_DISPONIBLE) * PENALIZACION_TIEMPO

    # Penalizo si me paso del espacio de bodega
    if volumen_total > ESPACIO_BODEGA:
        penalizacion += (volumen_total - ESPACIO_BODEGA) * PENALIZACION_VOLUMEN

    # Agrego los costos de setup por cambio entre modelos
    for l in LINEAS:
        sec = secuencia_por_linea[l]
        for i in range(1, len(sec)):
            anterior = sec[i - 1]
            actual = sec[i]
            if anterior in tiempo_setup and actual in tiempo_setup[anterior]:
                setup = tiempo_setup[anterior][actual]
                costo_total += setup * COSTO_SETUP_MIN

    # Devuelvo el costo total y la información de la solución
    total = costo_total + penalizacion

    return total, {
        "asignacion": asignacion_por_modelo,
        "secuencia": secuencia_por_linea,
        "individuo": individuo
    }

def cruzar(padre1, padre2):
    # Combino partes de dos padres para formar un nuevo individuo (corte aleatorio)
    corte = random.randint(1, len(padre1) - 2)
    hijo = padre1[:corte] + padre2[corte:]
    return hijo

def mutar(individuo, prob=0.1):
    # Modifico aleatoriamente la línea asignada de algunos modelos
    nuevo = individuo.copy()
    for i in range(len(nuevo)):
        if random.random() < prob:
            nuevo[i] = (nuevo[i][0], random.choice(LINEAS))
    return nuevo

def algoritmo_genetico(graficar_convergencia=False):
    # Inicializo la población con soluciones aleatorias
    poblacion = [crear_individuo() for _ in range(TAM_POBLACION)]

    historial_costos = []  # Aquí guardaré el mejor costo de cada generación

    # Itero por cada generación para ir mejorando las soluciones
    for gen in range(NUM_GENERACIONES):
        evaluados = [evaluar_individuo(ind) for ind in poblacion]
        evaluados.sort(key=lambda x: x[0])  # Ordeno por menor costo
        mejor_costo_gen = evaluados[0][0]
        historial_costos.append(mejor_costo_gen)

        # Selecciono la mitad superior (mejores soluciones) como padres
        poblacion = [ind["individuo"] for (_, ind) in evaluados[:TAM_POBLACION // 2]]

        nueva_poblacion = poblacion.copy()
        while len(nueva_poblacion) < TAM_POBLACION:
            # Selecciono padres al azar y genero hijos mutados
            p1, p2 = random.sample(poblacion, 2)
            hijo = cruzar(p1, p2)
            hijo = mutar(hijo)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    # El mejor individuo final
    mejor_costo, mejor_info = min([evaluar_individuo(ind) for ind in poblacion], key=lambda x: x[0])

    # Si se solicita, genero la gráfica de convergencia
    if graficar_convergencia:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, NUM_GENERACIONES + 1), historial_costos, linestyle='-', color='royalblue')
        plt.title("Convergencia del Algoritmo Genético")
        plt.xlabel("Generación")
        plt.ylabel("Mejor Costo Total")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mejor_costo, mejor_info


# ==========================
# MOSTRAR RESULTADOS
# ==========================

def mostrar_solucion(solucion, costo_total):
    print("\n===== MEJOR SOLUCIÓN =====")
    print(f"Costo total (producción + setup + penalización): ${round(costo_total, 2)}\n")

    print("Asignación de producción por modelo y línea:")
    for modelo in modelos:
        asign = solucion["asignacion"][modelo]
        total = sum(asign)
        print(f"  Modelo {modelo}: Total={total} → L1={asign[0]}, L2={asign[1]}, L3={asign[2]}")

    print("\nSecuencia de producción en cada línea:")
    for l in LINEAS:
        sec = solucion["secuencia"][l]
        sec_str = " → ".join(sec) if sec else "(vacía)"
        print(f"  Línea {l+1}: {sec_str}")

# ==========================
# EJECUCIÓN PRINCIPAL
# ==========================

if __name__ == "__main__":
    # Primero verifico si el problema tiene solución factible
    verificar_factibilidad()

    # Luego ejecuto el algoritmo genético para encontrar la mejor solución posible
    mejor_costo, mejor_solucion = algoritmo_genetico(graficar_convergencia=True)
    mostrar_solucion(mejor_solucion, mejor_costo)

    # Hago varias ejecuciones para comparar tiempos y resultados
    print("\n===== PRUEBA GENÉTICO: Múltiples ejecuciones con tiempo =====")
    for i in range(5):
        t0 = time.time()
        mejor_costo, _ = algoritmo_genetico()
        t1 = time.time()
        duracion = round(t1 - t0, 4)
        print(f"  Ejecución {i+1}: Costo total = ${round(mejor_costo, 2)}, Tiempo = {duracion}s")
