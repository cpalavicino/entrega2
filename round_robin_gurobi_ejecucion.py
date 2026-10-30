from itertools import combinations
from pathlib import Path

import pandas as pd
from gurobipy import GRB

from round_robin_gurobi_modelo import resolver


CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_INSTANCIAS = CARPETA_PROYECTO / "INSTANCIAS"
CARPETA_RESULTADOS = CARPETA_PROYECTO / "RESULTADOS"

# Completa este diccionario cuando se disponga del calendario real de otras ligas.
JORNADAS_CRITICAS_POR_INSTANCIA = {}

COLUMNAS_REQUERIDAS = {
    "Equipo",
    "Juega_Otras_Ligas",
    "ID_Estadio",
    "Peso_Audiencia",
}


def preparar_instancia(ruta_csv, jornadas_criticas=()):
    """Convierte una instancia CSV en los parámetros que requiere el modelo."""
    ruta_csv = Path(ruta_csv)
    df = pd.read_csv(ruta_csv)
    faltantes = COLUMNAS_REQUERIDAS.difference(df.columns)
    if faltantes:
        raise ValueError(
            f"{ruta_csv.name}: faltan columnas requeridas: {', '.join(sorted(faltantes))}."
        )
    if df.empty or len(df) % 2:
        raise ValueError(f"{ruta_csv.name}: el torneo debe tener un número par de equipos.")
    if df["Equipo"].isna().any() or df["Equipo"].duplicated().any():
        raise ValueError(f"{ruta_csv.name}: Equipo debe contener nombres únicos y no vacíos.")
    if df["ID_Estadio"].isna().any():
        raise ValueError(f"{ruta_csv.name}: ID_Estadio no puede contener valores vacíos.")

    juega_otras_ligas = pd.to_numeric(df["Juega_Otras_Ligas"], errors="raise")
    if not juega_otras_ligas.isin([0, 1]).all():
        raise ValueError(f"{ruta_csv.name}: Juega_Otras_Ligas solo puede contener 0 o 1.")
    peso_audiencia = pd.to_numeric(df["Peso_Audiencia"], errors="raise")

    equipos = df["Equipo"].astype(str).tolist()
    id_por_equipo = {equipo: i for i, equipo in enumerate(equipos, start=1)}
    cantidad_equipos = len(equipos)
    total_jornadas = cantidad_equipos - 1
    jornadas_criticas = sorted({int(jornada) for jornada in jornadas_criticas})
    if any(jornada < 1 or jornada > total_jornadas for jornada in jornadas_criticas):
        raise ValueError(
            f"{ruta_csv.name}: las jornadas críticas deben estar entre 1 y {total_jornadas}."
        )

    e_ext = [
        id_por_equipo[equipo]
        for equipo, participa in zip(equipos, juega_otras_ligas)
        if participa == 1
    ]

    pares_estadio = []
    for _, grupo in df.groupby("ID_Estadio", sort=False):
        ids = [id_por_equipo[equipo] for equipo in grupo["Equipo"].astype(str)]
        pares_estadio.extend(combinations(sorted(ids), 2))

    equipos_alta_audiencia = [
        id_por_equipo[equipo]
        for equipo, peso in zip(equipos, peso_audiencia)
        if peso == 100
    ]
    pares_alta_audiencia = list(combinations(sorted(equipos_alta_audiencia), 2))
    # En una jornada, cada equipo participa en un solo partido.
    max_partidos_alta_audiencia = len(equipos_alta_audiencia) // 2

    return {
        "n": cantidad_equipos // 2,
        "E_EXT": e_ext,
        "J_EXT": jornadas_criticas,
        "P": pares_alta_audiencia,
        "S": pares_estadio,
        "M_max": {jornada: max_partidos_alta_audiencia
                  for jornada in range(1, total_jornadas + 1)},
        "id_por_equipo": id_por_equipo,
        "nombre_instancia": ruta_csv.name,
    }


def resolver_instancia(
    ruta_csv,
    jornadas_criticas=(),
    w_b=1.0,
    w_ext=2.0,
    time_limit=3600,
    mip_gap=0.0,
    verbose=False,
    solver=None,
    estrategia="normal",
):
    """Lee, construye y resuelve una instancia; devuelve un resumen y el calendario."""
    if solver is None:
        solver = resolver
    if estrategia not in {"normal", "sin_cortes_heuristicas"}:
        raise ValueError(
            "estrategia debe ser 'normal' o 'sin_cortes_heuristicas'."
        )

    datos = preparar_instancia(ruta_csv, jornadas_criticas)
    parametros_modelo = {
        clave: datos[clave]
        for clave in ("n", "E_EXT", "J_EXT", "P", "S", "M_max")
    }
    modelo, calendario = solver(
        **parametros_modelo,
        w_b=w_b,
        w_ext=w_ext,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
        desactivar_cortes_heuristicas=(estrategia == "sin_cortes_heuristicas"),
    )

    nombres = {identificador: equipo for equipo, identificador in datos["id_por_equipo"].items()}
    partidos = [
        {
            "Jornada": jornada,
            "Local": nombres[local],
            "Visitante": nombres[visitante],
        }
        for jornada, encuentros in calendario.items()
        for local, visitante in encuentros
    ]
    if modelo.SolCount:
        if modelo.Status == GRB.OPTIMAL:
            estado = "Óptimo"
        else:
            estado = "Solución factible (no se probó optimalidad)"
        objetivo = modelo.ObjVal
    else:
        estado = (
            "Infactible"
            if modelo.Status == GRB.INFEASIBLE
            else f"Sin solución (estado {modelo.Status})"
        )
        objetivo = None
    cota_inferior = (
        None
        if modelo.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}
        else modelo.ObjBoundC
    )
    brecha = modelo.MIPGap if modelo.SolCount else None

    return {
        "Instancia": datos["nombre_instancia"],
        "Estrategia": estrategia,
        "Equipos": 2 * datos["n"],
        "Jornadas": 2 * datos["n"] - 1,
        "Estado": estado,
        "ObjVal": objetivo,
        "ObjBoundC": cota_inferior,
        "MIPGap": brecha,
        "Runtime (s)": modelo.Runtime,
        "NodeCount": modelo.NodeCount,
        "Equipos otras ligas": len(datos["E_EXT"]),
        "Jornadas críticas": len(datos["J_EXT"]),
        "Pares estadio": len(datos["S"]),
        "Pares alta audiencia": len(datos["P"]),
        "Calendario": partidos,
    }


def guardar_calendario(resultado, carpeta_salida):
    """Guarda el calendario de una instancia resuelta como CSV."""
    if resultado["ObjVal"] is None:
        return None

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    nombre = Path(resultado["Instancia"]).stem.replace("Instancia_Sintetica_", "")
    ruta_salida = carpeta_salida / (
        f"Calendario_Resuelto_{nombre}_{resultado['Estrategia']}.csv"
    )
    pd.DataFrame(resultado["Calendario"]).to_csv(ruta_salida, index=False)
    return ruta_salida


def ejecutar_todas_instancias(
    carpeta_instancias=CARPETA_INSTANCIAS,
    carpeta_resultados=CARPETA_RESULTADOS,
    jornadas_criticas_por_instancia=None,
    w_b=1.0,
    w_ext=1.0,
    time_limit=3600,
    mip_gap=0.0,
    verbose=False,
    guardar_reporte=True,
):
    """Resuelve secuencialmente todos los CSV de una carpeta y guarda sus calendarios."""
    carpeta_instancias = Path(carpeta_instancias)
    archivos_instancia = sorted(carpeta_instancias.glob("*.csv"))
    if not archivos_instancia:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV de instancias en {carpeta_instancias}."
        )

    if jornadas_criticas_por_instancia is None:
        jornadas_criticas_por_instancia = JORNADAS_CRITICAS_POR_INSTANCIA

    resultados = []
    for ruta_csv in archivos_instancia:
        for estrategia in ("normal", "sin_cortes_heuristicas"):
            print(f"Resolviendo {ruta_csv.name} ({estrategia})...")
            resultado = resolver_instancia(
                ruta_csv,
                jornadas_criticas=jornadas_criticas_por_instancia.get(ruta_csv.name, ()),
                w_b=w_b,
                w_ext=w_ext,
                time_limit=time_limit,
                mip_gap=mip_gap,
                verbose=verbose,
                estrategia=estrategia,
            )
            ruta_calendario = guardar_calendario(resultado, carpeta_resultados)
            if ruta_calendario is None:
                print(f"Estado: {resultado['Estado']} | no se obtuvo calendario factible.")
            else:
                print(f"Estado: {resultado['Estado']} | ObjVal: {resultado['ObjVal']:.2f}")
                print(f"Calendario guardado en: {ruta_calendario}")
            resultados.append(resultado)

    resumen = pd.DataFrame(
        [
            {clave: valor for clave, valor in resultado.items() if clave != "Calendario"}
            for resultado in resultados
        ]
    )
    if guardar_reporte:
        Path(carpeta_resultados).mkdir(parents=True, exist_ok=True)
        ruta_reporte = Path(carpeta_resultados) / "Comparacion_Estrategias_Gurobi.csv"
        resumen.to_csv(ruta_reporte, index=False)
        print(f"Reporte de comparación guardado en: {ruta_reporte}")

    return resumen, resultados


if __name__ == "__main__":
    resumen_global, _ = ejecutar_todas_instancias()
    print("\nRESUMEN GLOBAL DE OPTIMIZACIÓN")
    print(resumen_global.to_string(index=False))
