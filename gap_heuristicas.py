import csv
import hashlib
import math
import random
from pathlib import Path


ESTRATEGIAS_GUROBI = ("normal", "sin_cortes_heuristicas")


def seleccionar_jornadas_criticas(
    nombre_instancia,
    cantidad_equipos,
    cantidad_equipos_externos,
    factor=0.40,
    semilla=42,
):
    cantidad_jornadas = cantidad_equipos - 1
    cantidad_criticas = math.ceil(
        cantidad_jornadas * factor * cantidad_equipos_externos / cantidad_equipos
    )
    semilla_instancia = hashlib.sha256(
        f"{semilla}:{Path(nombre_instancia).name}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(semilla_instancia[:8], "big"))
    return sorted(rng.sample(range(1, cantidad_jornadas + 1), cantidad_criticas))


def cargar_referencias_gurobi(ruta_reporte, instancias_requeridas=()):
    ruta_reporte = Path(ruta_reporte)
    if not ruta_reporte.is_file():
        raise FileNotFoundError(
            f"No se encontró {ruta_reporte}. Ejecute primero "
            "round_robin_gurobi_ejecucion.py para generar la referencia global."
        )

    referencias = {}
    pesos = set()
    with ruta_reporte.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        columnas_requeridas = {
            "Instancia",
            "Estrategia",
            "ObjVal",
            "Peso_breaks",
            "Peso_otras_ligas",
        }
        faltantes = columnas_requeridas.difference(lector.fieldnames or [])
        if faltantes:
            raise ValueError(
                f"{ruta_reporte.name}: faltan columnas requeridas: "
                f"{', '.join(sorted(faltantes))}. Vuelva a ejecutar la versión "
                "global actualizada para guardar los pesos del objetivo."
            )

        for fila in lector:
            instancia = fila["Instancia"]
            estrategia = fila["Estrategia"]
            if estrategia not in ESTRATEGIAS_GUROBI:
                raise ValueError(
                    f"{ruta_reporte.name}: estrategia desconocida "
                    f"{estrategia!r} para {instancia}."
                )
            try:
                w_b = float(fila["Peso_breaks"])
                w_ext = float(fila["Peso_otras_ligas"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{ruta_reporte.name}: pesos inválidos para {instancia}."
                ) from error
            if not math.isfinite(w_b) or not math.isfinite(w_ext):
                raise ValueError(
                    f"{ruta_reporte.name}: pesos no finitos para {instancia}."
                )
            pesos.add((w_b, w_ext))

            if instancia not in referencias:
                referencias[instancia] = {}
            if estrategia in referencias[instancia]:
                raise ValueError(
                    f"{ruta_reporte.name}: hay más de un resultado para "
                    f"{instancia} ({estrategia})."
                )
            valor_objetivo = fila["ObjVal"].strip()
            try:
                objetivo = float(valor_objetivo) if valor_objetivo else None
            except ValueError as error:
                raise ValueError(
                    f"{ruta_reporte.name}: ObjVal inválido para "
                    f"{instancia} ({estrategia})."
                ) from error
            if objetivo is not None and not math.isfinite(objetivo):
                raise ValueError(
                    f"{ruta_reporte.name}: ObjVal no finito para "
                    f"{instancia} ({estrategia})."
                )
            referencias[instancia][estrategia] = objetivo

    if len(pesos) != 1:
        raise ValueError(
            f"{ruta_reporte.name}: todas las estrategias deben usar los mismos "
            "pesos de objetivo."
        )
    if not referencias:
        raise ValueError(f"{ruta_reporte.name}: el reporte no contiene resultados.")

    for instancia in instancias_requeridas:
        estrategias_disponibles = referencias.get(Path(instancia).name, {})
        faltantes = set(ESTRATEGIAS_GUROBI).difference(estrategias_disponibles)
        if faltantes:
            raise ValueError(
                f"{ruta_reporte.name}: faltan referencias para {Path(instancia).name}: "
                f"{', '.join(sorted(faltantes))}. Complete la ejecución global."
            )

    w_b, w_ext = next(iter(pesos))
    return {"resultados": referencias, "w_b": w_b, "w_ext": w_ext}


def calcular_gap_porcentual(objetivo_heuristica, objetivo_gurobi):
    if objetivo_heuristica is None or objetivo_gurobi is None or objetivo_gurobi == 0:
        return None
    return 100 * (objetivo_heuristica - objetivo_gurobi) / abs(objetivo_gurobi)
