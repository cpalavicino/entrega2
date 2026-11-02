import pandas as pd
import numpy as np
import random
import time
import math
from pathlib import Path

# ==========================================
# 1. LECTURA DE INSTANCIAS Y PREPARACIÓN
# ==========================================
def cargar_instancia(filepath, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv(filepath)
    N = len(df)
    jornadas_totales = N - 1
    
    # Conjunto E_EXT: Equipos que juegan otras ligas
    e_ext = df[df['Juega_Otras_Ligas'] == 1].index.tolist()
    
    # Jornadas Críticas J_EXT (Regla del 40%)
    porcentaje_critico = 0.40 * (len(e_ext) / N)
    num_jornadas_criticas = math.ceil(porcentaje_critico * jornadas_totales)
    j_ext = random.sample(range(jornadas_totales), num_jornadas_criticas)
    
    # Conjunto S: Estadios compartidos
    # Agrupamos por ID_Estadio y guardamos pares que lo comparten
    s_estadios = []
    estadios_agrupados = df.groupby('ID_Estadio').groups
    for estadio, indices in estadios_agrupados.items():
        if len(indices) > 1:
            # Si hay más de un equipo en este estadio, creamos los pares
            equipos_comparten = list(indices)
            for i in range(len(equipos_comparten)):
                for j in range(i+1, len(equipos_comparten)):
                    s_estadios.append((equipos_comparten[i], equipos_comparten[j]))
                    
    # Conjunto P: Partidos de alta demanda (Clásicos)
    # Definiremos un clásico si AMBOS equipos tienen Peso_Audiencia == 100
    p_clasicos = []
    equipos_top = df[df['Peso_Audiencia'] >= 100].index.tolist()
    for i in range(len(equipos_top)):
        for j in range(i+1, len(equipos_top)):
            p_clasicos.append((equipos_top[i], equipos_top[j]))
            
    return N, jornadas_totales, e_ext, j_ext, s_estadios, p_clasicos


# ==========================================
# 2. SOLUCIÓN INICIAL (Método del Polígono)
# ==========================================
def generar_solucion_inicial(N):
    S = np.zeros((N, N - 1), dtype=int)
    equipos = list(range(N))
    fijo = equipos[0]
    moviles = equipos[1:]
    
    for t in range(N - 1):
        # Emparejar el fijo con el primer movil
        if t % 2 == 0:
            S[fijo, t] = moviles[0] + 1
            S[moviles[0], t] = -(fijo + 1)
        else:
            S[fijo, t] = -(moviles[0] + 1)
            S[moviles[0], t] = fijo + 1
            
        # Emparejar el resto
        for i in range(1, N // 2):
            eq1 = moviles[i]
            eq2 = moviles[N - 1 - i]
            if (t + i) % 2 == 0:
                S[eq1, t] = eq2 + 1
                S[eq2, t] = -(eq1 + 1)
            else:
                S[eq1, t] = -(eq2 + 1)
                S[eq2, t] = eq1 + 1
                
        # Rotar los móviles
        moviles.insert(0, moviles.pop())
        
    return S


# ==========================================
# 3. EVALUACIÓN DE FITNESS (Función Penalizada)
# ==========================================
def evaluar_fitness(S, e_ext, j_ext, s_estadios, p_clasicos, Mmax=1, w_b=1, w_ext=10, P_TV=10000, P_EST=10000):
    N, jornadas_totales = S.shape
    fitness = 0
    
    # 1. Contar Breaks y Penalizar 3 locales consecutivos
    for i in range(N):
        # Cuenta breaks normales (2 seguidos)
        for t in range(jornadas_totales - 1):
            if np.sign(S[i, t]) == np.sign(S[i, t+1]):
                fitness += w_b
                
        # NUEVO: Penaliza estrictamente 3 partidos de local consecutivos
        for t in range(jornadas_totales - 2):
            if S[i, t] > 0 and S[i, t+1] > 0 and S[i, t+2] > 0:
                fitness += 10000
                
    # 2. Viajes Externos Críticos
    for i in e_ext:
        for t in j_ext:
            if S[i, t] < 0: # Si es visitante
                fitness += w_ext
                
    # 3. Restricciones por Jornada (TV y Estadios)
    for t in range(jornadas_totales):
        # Clásicos
        clasicos_fecha = 0
        for (eq1, eq2) in p_clasicos:
            if abs(S[eq1, t]) == eq2 + 1:
                clasicos_fecha += 1
        if clasicos_fecha > Mmax:
            fitness += P_TV * (clasicos_fecha - Mmax)
            
        # Estadios compartidos
        for (eq1, eq2) in s_estadios:
            if S[eq1, t] > 0 and S[eq2, t] > 0:
                fitness += P_EST
                
    return fitness


# ==========================================
# 4. BÚSQUEDA LOCAL (Swap de Jornadas)
# ==========================================
def busqueda_local(S, e_ext, j_ext, s_estadios, p_clasicos, Mmax):
    N, jornadas_totales = S.shape
    S_best = S.copy()
    f_best = evaluar_fitness(S_best, e_ext, j_ext, s_estadios, p_clasicos, Mmax)
    
    mejora = True
    while mejora:
        mejora = False
        for i in range(jornadas_totales - 1):
            for j in range(i + 1, jornadas_totales):
                S_temp = S_best.copy()
                # Swap de columnas completas
                S_temp[:, [i, j]] = S_temp[:, [j, i]]
                
                f_temp = evaluar_fitness(S_temp, e_ext, j_ext, s_estadios, p_clasicos, Mmax)
                if f_temp < f_best:
                    S_best = S_temp
                    f_best = f_temp
                    mejora = True
                    break # Primera mejora
            if mejora:
                break
    return S_best, f_best


# ==========================================
# 5. PERTURBACIÓN
# ==========================================
def perturbacion(S):
    S_prima = S.copy()
    jornadas_totales = S.shape[1]
    
    k = random.randint(2, 4) # Intercambios bruscos
    for _ in range(k):
        a, b = random.sample(range(jornadas_totales), 2)
        S_prima[:, [a, b]] = S_prima[:, [b, a]]
        
    return S_prima


# ==========================================
# 6. CICLO PRINCIPAL ILS
# ==========================================
def ejecutar_ils(filepath, Mmax=1, max_iter=200, time_limit=300, seed=42):
    start_time = time.time()
    
    # Cargar datos
    N, jornadas_totales, e_ext, j_ext, s_estadios, p_clasicos = cargar_instancia(
        filepath, seed=seed
    )
    
    # Inicialización
    S_actual = generar_solucion_inicial(N)
    S_actual, f_actual = busqueda_local(S_actual, e_ext, j_ext, s_estadios, p_clasicos, Mmax)
    
    S_mejor = S_actual.copy()
    f_mejor = f_actual
    
    iter_sin_mejora = 0
    iteracion = 0
    
    while iter_sin_mejora < max_iter and (time.time() - start_time) < time_limit:
        iteracion += 1
        
        # Perturbación y nueva Búsqueda Local
        S_prima = perturbacion(S_actual)
        S_candidata, f_candidata = busqueda_local(S_prima, e_ext, j_ext, s_estadios, p_clasicos, Mmax)
        
        # Actualizar mejor histórico
        if f_candidata < f_mejor:
            S_mejor = S_candidata.copy()
            f_mejor = f_candidata
            iter_sin_mejora = 0
        else:
            iter_sin_mejora += 1
            
        # Criterio de Aceptación (5% de tolerancia)
        if f_candidata <= f_mejor * 1.05:
            S_actual = S_candidata.copy()
            f_actual = f_candidata
        else:
            S_actual = S_mejor.copy()
            f_actual = f_mejor
            
    tiempo_total = time.time() - start_time
    
    return S_mejor, f_mejor, iteracion, tiempo_total, sorted(j_ext)


def ejecutar_todas_instancias(
    carpeta_instancias=None,
    carpeta_resultados=None,
    semillas=range(1, 6),
    Mmax=1,
    max_iter=500,
    time_limit=600,
):
    """Ejecuta cada CSV cinco veces (o con las semillas indicadas) y guarda el reporte."""
    carpeta_proyecto = Path(__file__).resolve().parents[1]
    carpeta_instancias = Path(carpeta_instancias or carpeta_proyecto / "INSTANCIAS")
    carpeta_resultados = Path(carpeta_resultados or carpeta_proyecto / "RESULTADOS")
    archivos_instancia = sorted(carpeta_instancias.glob("*.csv"))
    semillas = tuple(semillas)

    if not archivos_instancia:
        raise FileNotFoundError(
            f"No se encontraron instancias CSV en {carpeta_instancias}."
        )
    if not semillas:
        raise ValueError("Debe proporcionarse al menos una semilla.")

    resultados = []
    for archivo in archivos_instancia:
        for seed in semillas:
            print(f"Ejecutando ILS: {archivo.name} | semilla={seed}")
            _, fitness, iteraciones, tiempo, jornadas_criticas = ejecutar_ils(
                archivo,
                Mmax=Mmax,
                max_iter=max_iter,
                time_limit=time_limit,
                seed=seed,
            )
            resultado = {
                "Instancia": archivo.name,
                "Semilla": seed,
                "Equipos": len(pd.read_csv(archivo)),
                "Jornadas_criticas": jornadas_criticas,
                "Fitness_ILS": fitness,
                "Iteraciones": iteraciones,
                "Tiempo_s": tiempo,
            }
            resultados.append(resultado)
            print(
                f"Fitness={fitness:.2f} | iteraciones={iteraciones} "
                f"| tiempo={tiempo:.2f}s"
            )

    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    ruta_reporte = carpeta_resultados / "Resultados_ILS_5_semillas.csv"
    pd.DataFrame(resultados).to_csv(ruta_reporte, index=False)
    print(f"Reporte guardado en: {ruta_reporte}")
    return pd.DataFrame(resultados)


if __name__ == "__main__":
    ejecutar_todas_instancias()
