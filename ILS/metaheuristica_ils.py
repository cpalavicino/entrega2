import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gap_heuristicas import (
    calcular_gap_porcentual,
    cargar_referencias_gurobi,
    seleccionar_jornadas_criticas,
)

# ==========================================
# 1. LECTURA DE INSTANCIAS Y PREPARACIÓN
# ==========================================
def cargar_instancia(filepath, seed=42, critical_seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv(filepath)
    N = len(df)
    jornadas_totales = N - 1
    
    # Conjunto E_EXT: Equipos que juegan otras ligas
    e_ext = df[df['Juega_Otras_Ligas'] == 1].index.tolist()
    
    # Jornadas Críticas J_EXT (Regla del 40%)
    j_ext = [
        jornada - 1
        for jornada in seleccionar_jornadas_criticas(
            Path(filepath).name,
            N,
            len(e_ext),
            factor=0.40,
            semilla=critical_seed,
        )
    ]
    
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
    equipos_top = df[df['Peso_Audiencia'] == 100].index.tolist()
    for i in range(len(equipos_top)):
        for j in range(i+1, len(equipos_top)):
            p_clasicos.append((equipos_top[i], equipos_top[j]))
            
    max_clasicos_por_jornada = len(equipos_top) // 2
    return (
        N,
        jornadas_totales,
        e_ext,
        j_ext,
        s_estadios,
        p_clasicos,
        max_clasicos_por_jornada,
    )


# ==========================================
# 2. SOLUCIÓN INICIAL (Método del Polígono)
# ==========================================
def generar_solucion_inicial(N):
    if N < 2 or N % 2:
        raise ValueError("La cantidad de equipos debe ser par y al menos 2.")

    S = np.zeros((N, N - 1), dtype=int)
    equipos = list(range(N))
    cantidad_por_lado = N // 2

    for t in range(N - 1):
        for i in range(cantidad_por_lado):
            equipo_a = equipos[i]
            equipo_b = equipos[N - 1 - i]
            distancia = (equipo_b - equipo_a) % N
            if distancia == cantidad_por_lado:
                local = min(equipo_a, equipo_b)
            elif distancia < cantidad_por_lado:
                local = equipo_a
            else:
                local = equipo_b
            visitante = equipo_b if local == equipo_a else equipo_a
            S[local, t] = visitante + 1
            S[visitante, t] = -(local + 1)

        equipos = [equipos[0], equipos[-1], *equipos[1:-1]]

    return S


# ==========================================
# 3. EVALUACIÓN DE FITNESS (Función Penalizada)
# ==========================================
def evaluar_estado(
    S,
    e_ext,
    j_ext,
    s_estadios,
    p_clasicos,
    Mmax,
    w_b,
    w_ext,
    verificar_estructura=True,
):
    """Devuelve el número de restricciones incumplidas y el objetivo del modelo."""
    N, jornadas_totales = S.shape
    if jornadas_totales != N - 1 or N % 2:
        raise ValueError("El calendario debe tener N equipos y N-1 jornadas.")

    violaciones = 0
    cantidad_localias = np.sum(S > 0, axis=1)
    minimo_localias = N // 2 - 1
    maximo_localias = N // 2
    violaciones += int(
        np.maximum(minimo_localias - cantidad_localias, 0).sum()
        + np.maximum(cantidad_localias - maximo_localias, 0).sum()
    )

    if verificar_estructura:
        pares_enfrentados = {}
        for jornada in range(jornadas_totales):
            for equipo in range(N):
                valor = int(S[equipo, jornada])
                rival = abs(valor) - 1
                if valor == 0 or rival < 0 or rival >= N or rival == equipo:
                    violaciones += 1
                    continue
                if int(S[rival, jornada]) != (-equipo - 1 if valor > 0 else equipo + 1):
                    violaciones += 1
                if equipo < rival:
                    par = (equipo, rival)
                    pares_enfrentados[par] = pares_enfrentados.get(par, 0) + 1
        for equipo_a in range(N):
            for equipo_b in range(equipo_a + 1, N):
                veces = pares_enfrentados.get((equipo_a, equipo_b), 0)
                if veces != 1:
                    violaciones += abs(veces - 1)

    breaks = np.count_nonzero(
        ((S[:, :-1] > 0) & (S[:, 1:] > 0))
        | ((S[:, :-1] < 0) & (S[:, 1:] < 0))
    )
    visitas_criticas = 0
    if e_ext and j_ext:
        visitas_criticas = np.count_nonzero(S[np.ix_(e_ext, j_ext)] < 0)

    if p_clasicos:
        equipos_a, equipos_b = np.asarray(p_clasicos).T
        clasicos_por_jornada = np.count_nonzero(
            np.abs(S[equipos_a, :]) == (equipos_b + 1)[:, np.newaxis],
            axis=0,
        )
        violaciones += int(np.maximum(clasicos_por_jornada - Mmax, 0).sum())

    if s_estadios:
        equipos_a, equipos_b = np.asarray(s_estadios).T
        conflictos_por_jornada = np.count_nonzero(
            (S[equipos_a, :] > 0) & (S[equipos_b, :] > 0),
            axis=0,
        )
        violaciones += int(conflictos_por_jornada.sum())

    if jornadas_totales >= 3:
        tres_localias = (
            (S[:, :-2] > 0) & (S[:, 1:-1] > 0) & (S[:, 2:] > 0)
        )
        tres_visitas = (
            (S[:, :-2] < 0) & (S[:, 1:-1] < 0) & (S[:, 2:] < 0)
        )
        violaciones += int(np.count_nonzero(tres_localias | tres_visitas))

    objetivo = w_b * breaks + w_ext * visitas_criticas
    return violaciones, objetivo


def evaluar_fitness(
    S,
    e_ext,
    j_ext,
    s_estadios,
    p_clasicos,
    Mmax=1,
    w_b=1,
    w_ext=2,
    P_TV=10000,
    P_EST=10000,
    verificar_estructura=True,
):
    """Prioriza calendarios factibles y luego minimiza el objetivo del modelo."""
    violaciones, objetivo = evaluar_estado(
        S,
        e_ext,
        j_ext,
        s_estadios,
        p_clasicos,
        Mmax,
        w_b,
        w_ext,
        verificar_estructura=verificar_estructura,
    )
    N, jornadas_totales = S.shape
    max_objetivo = (
        abs(w_b) * N * max(0, jornadas_totales - 1)
        + abs(w_ext) * len(e_ext) * len(j_ext)
    )
    penalizacion = max(P_TV, P_EST, max_objetivo + 1)
    return objetivo + penalizacion * violaciones


# ==========================================
# 4. BÚSQUEDA LOCAL (Swap de Jornadas)
# ==========================================
def busqueda_local(
    S,
    e_ext,
    j_ext,
    s_estadios,
    p_clasicos,
    Mmax,
    w_b=1.0,
    w_ext=2.0,
):
    N, jornadas_totales = S.shape
    S_best = S.copy()
    f_best = evaluar_fitness(
        S_best,
        e_ext,
        j_ext,
        s_estadios,
        p_clasicos,
        Mmax,
        w_b=w_b,
        w_ext=w_ext,
        verificar_estructura=False,
    )

    mejora = True
    while mejora:
        mejora = False
        for i in range(jornadas_totales - 1):
            for j in range(i + 1, jornadas_totales):
                S_temp = S_best.copy()
                # Swap de columnas completas
                S_temp[:, [i, j]] = S_temp[:, [j, i]]
                
                f_temp = evaluar_fitness(
                    S_temp,
                    e_ext,
                    j_ext,
                    p_clasicos=p_clasicos,
                    s_estadios=s_estadios,
                    Mmax=Mmax,
                    w_b=w_b,
                    w_ext=w_ext,
                    verificar_estructura=False,
                )
                if f_temp < f_best:
                    S_best = S_temp
                    f_best = f_temp
                    mejora = True
                    break # Primera mejora
            if mejora:
                break
        if mejora:
            continue

        for jornada in range(jornadas_totales):
            S_temp = S_best.copy()
            S_temp[:, jornada] *= -1
            f_temp = evaluar_fitness(
                S_temp,
                e_ext,
                j_ext,
                s_estadios,
                p_clasicos,
                Mmax,
                w_b=w_b,
                w_ext=w_ext,
                verificar_estructura=False,
            )
            if f_temp < f_best:
                S_best = S_temp
                f_best = f_temp
                mejora = True
                break
        if mejora:
            continue

        minimo_localias = N // 2 - 1
        maximo_localias = N // 2
        cantidad_localias = np.sum(S_best > 0, axis=1)
        for jornada in range(jornadas_totales):
            for local in np.flatnonzero(S_best[:, jornada] > 0):
                visitante = abs(int(S_best[local, jornada])) - 1
                if (
                    cantidad_localias[local] <= minimo_localias
                    or cantidad_localias[visitante] >= maximo_localias
                ):
                    continue
                S_temp = S_best.copy()
                S_temp[local, jornada] *= -1
                S_temp[visitante, jornada] *= -1
                f_temp = evaluar_fitness(
                    S_temp,
                    e_ext,
                    j_ext,
                    s_estadios,
                    p_clasicos,
                    Mmax,
                    w_b=w_b,
                    w_ext=w_ext,
                    verificar_estructura=False,
                )
                if f_temp < f_best:
                    S_best = S_temp
                    f_best = f_temp
                    mejora = True
                    break
            if mejora:
                break
        if mejora:
            continue

        for jornada_conflicto in range(jornadas_totales):
            for equipo_a, equipo_b in s_estadios:
                if not (
                    S_best[equipo_a, jornada_conflicto] > 0
                    and S_best[equipo_b, jornada_conflicto] > 0
                ):
                    continue
                for equipo in (equipo_a, equipo_b):
                    visitante_conflicto = abs(
                        int(S_best[equipo, jornada_conflicto])
                    ) - 1
                    if cantidad_localias[visitante_conflicto] >= maximo_localias:
                        continue
                    for jornada_away in np.flatnonzero(S_best[equipo, :] < 0):
                        donante = abs(int(S_best[equipo, jornada_away])) - 1
                        if cantidad_localias[donante] <= minimo_localias:
                            continue
                        S_temp = S_best.copy()
                        S_temp[equipo, jornada_conflicto] *= -1
                        S_temp[visitante_conflicto, jornada_conflicto] *= -1
                        S_temp[equipo, jornada_away] *= -1
                        S_temp[donante, jornada_away] *= -1
                        f_temp = evaluar_fitness(
                            S_temp,
                            e_ext,
                            j_ext,
                            s_estadios,
                            p_clasicos,
                            Mmax,
                            w_b=w_b,
                            w_ext=w_ext,
                            verificar_estructura=False,
                        )
                        if f_temp < f_best:
                            S_best = S_temp
                            f_best = f_temp
                            mejora = True
                            break
                    if mejora:
                        break
                if mejora:
                    break
    return S_best, f_best


# ==========================================
# 5. PERTURBACIÓN
# ==========================================
def perturbacion(S):
    S_prima = S.copy()
    jornadas_totales = S.shape[1]
    cantidad_equipos = S.shape[0]
    minimo_localias = cantidad_equipos // 2 - 1
    maximo_localias = cantidad_equipos // 2

    k = random.randint(2, 4)
    for _ in range(k):
        movimiento = random.choice(("intercambiar", "invertir_jornada", "cambiar_localia"))
        if movimiento == "intercambiar":
            a, b = random.sample(range(jornadas_totales), 2)
            S_prima[:, [a, b]] = S_prima[:, [b, a]]
        elif movimiento == "invertir_jornada":
            jornada = random.randrange(jornadas_totales)
            S_prima[:, jornada] *= -1
        else:
            cantidad_localias = np.sum(S_prima > 0, axis=1)
            movimientos_posibles = []
            for jornada in range(jornadas_totales):
                for local in np.flatnonzero(S_prima[:, jornada] > 0):
                    visitante = abs(int(S_prima[local, jornada])) - 1
                    if (
                        cantidad_localias[local] > minimo_localias
                        and cantidad_localias[visitante] < maximo_localias
                    ):
                        movimientos_posibles.append((local, visitante, jornada))
            if movimientos_posibles:
                local, visitante, jornada = random.choice(movimientos_posibles)
                S_prima[local, jornada] *= -1
                S_prima[visitante, jornada] *= -1
            else:
                jornada = random.randrange(jornadas_totales)
                S_prima[:, jornada] *= -1

    return S_prima


# ==========================================
# 6. CICLO PRINCIPAL ILS
# ==========================================
def ejecutar_ils(
    filepath,
    Mmax=None,
    max_iter=500,
    seed=42,
    w_b=1.0,
    w_ext=2.0,
):
    start_time = time.time()
    
    # Cargar datos
    (
        N,
        jornadas_totales,
        e_ext,
        j_ext,
        s_estadios,
        p_clasicos,
        mmax_instancia,
    ) = cargar_instancia(
        filepath, seed=seed
    )
    if Mmax is None:
        Mmax = mmax_instancia
    
    # Inicialización
    S_actual = generar_solucion_inicial(N)
    S_actual, f_actual = busqueda_local(
        S_actual,
        e_ext,
        j_ext,
        s_estadios,
        p_clasicos,
        Mmax,
        w_b=w_b,
        w_ext=w_ext,
    )
    
    S_mejor = S_actual.copy()
    f_mejor = f_actual
    violaciones_mejor, objetivo_mejor = evaluar_estado(
        S_mejor,
        e_ext,
        j_ext,
        s_estadios,
        p_clasicos,
        Mmax,
        w_b,
        w_ext,
        verificar_estructura=False,
    )
    
    iter_sin_mejora = 0
    iteracion = 0
    
    while iter_sin_mejora < max_iter:
        iteracion += 1
        
        # Perturbación y nueva Búsqueda Local
        S_prima = perturbacion(S_actual)
        S_candidata, f_candidata = busqueda_local(
            S_prima,
            e_ext,
            j_ext,
            s_estadios,
            p_clasicos,
            Mmax,
            w_b=w_b,
            w_ext=w_ext,
        )
        violaciones_candidata, objetivo_candidata = evaluar_estado(
            S_candidata,
            e_ext,
            j_ext,
            s_estadios,
            p_clasicos,
            Mmax,
            w_b,
            w_ext,
            verificar_estructura=False,
        )
        
        # Actualizar mejor histórico
        if (
            violaciones_candidata < violaciones_mejor
            or (
                violaciones_candidata == violaciones_mejor
                and objetivo_candidata < objetivo_mejor
            )
        ):
            S_mejor = S_candidata.copy()
            f_mejor = f_candidata
            violaciones_mejor = violaciones_candidata
            objetivo_mejor = objetivo_candidata
            iter_sin_mejora = 0
        else:
            iter_sin_mejora += 1
            
        # La tolerancia se aplica al objetivo, nunca a la factibilidad.
        if (
            violaciones_candidata < violaciones_mejor
            or (
                violaciones_candidata == violaciones_mejor
                and objetivo_candidata <= objetivo_mejor * 1.05
            )
        ):
            S_actual = S_candidata.copy()
            f_actual = f_candidata
        else:
            S_actual = S_mejor.copy()
            f_actual = f_mejor
            
    tiempo_total = time.time() - start_time
    
    return S_mejor, f_mejor, iteracion, tiempo_total, sorted(jornada + 1 for jornada in j_ext)


def evaluar_objetivo_comparable(
    S,
    e_ext,
    j_ext,
    s_estadios,
    p_clasicos,
    Mmax,
    w_b,
    w_ext,
):
    """Evalúa una solución con el objetivo y las restricciones del modelo Gurobi."""
    violaciones, objetivo = evaluar_estado(
        S, e_ext, j_ext, s_estadios, p_clasicos, Mmax, w_b, w_ext
    )
    return objetivo, violaciones == 0


def ejecutar_todas_instancias(
    carpeta_instancias=None,
    carpeta_resultados=None,
    semillas=range(1, 6),
    Mmax=None,
    max_iter=500,
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

    referencias = cargar_referencias_gurobi(
        carpeta_resultados / "Comparacion_Estrategias_Gurobi.csv",
        [archivo.name for archivo in archivos_instancia],
    )
    resultados = []
    for archivo in archivos_instancia:
        (
            _,
            _,
            e_ext,
            j_ext,
            s_estadios,
            p_clasicos,
            mmax_instancia,
        ) = cargar_instancia(archivo)
        mmax_ejecucion = mmax_instancia if Mmax is None else Mmax
        referencia_instancia = referencias["resultados"][archivo.name]
        for seed in semillas:
            print(f"Ejecutando ILS: {archivo.name} | semilla={seed}")
            solucion, fitness, iteraciones, tiempo, jornadas_criticas = ejecutar_ils(
                archivo,
                Mmax=mmax_ejecucion,
                max_iter=max_iter,
                seed=seed,
                w_b=referencias["w_b"],
                w_ext=referencias["w_ext"],
            )
            objetivo_comparable, factible_comparable = evaluar_objetivo_comparable(
                solucion,
                e_ext,
                j_ext,
                s_estadios,
                p_clasicos,
                mmax_instancia,
                referencias["w_b"],
                referencias["w_ext"],
            )
            resultado = {
                "Instancia": archivo.name,
                "Semilla": seed,
                "Equipos": len(pd.read_csv(archivo)),
                "Jornadas_criticas": jornadas_criticas,
                "Fitness_ILS": fitness,
                "Objetivo_comparable_Gurobi": objetivo_comparable,
                "Factible_comparable_Gurobi": factible_comparable,
                "ObjVal_Gurobi_normal": referencia_instancia["normal"],
                "Gap_normal_%": (
                    calcular_gap_porcentual(
                        objetivo_comparable, referencia_instancia["normal"]
                    )
                    if factible_comparable
                    else None
                ),
                "ObjVal_Gurobi_sin_cortes_heuristicas":
                    referencia_instancia["sin_cortes_heuristicas"],
                "Gap_sin_cortes_heuristicas_%": (
                    calcular_gap_porcentual(
                        objetivo_comparable,
                        referencia_instancia["sin_cortes_heuristicas"],
                    )
                    if factible_comparable
                    else None
                ),
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
