
import gurobipy as gp
from gurobipy import GRB


def resolver(n, E_EXT=(), J_EXT=(), P=(), S=(), M_max=0, w_b=1.0, w_ext=1.0,
             time_limit=3600, mip_gap=0.0, verbose=True,
             desactivar_cortes_heuristicas=False):
    """
    n      : mitad del número de equipos (equipos 1..2n, jornadas 1..2n-1)
    E_EXT  : equipos que participan en otras ligas
    J_EXT  : jornadas críticas de otras ligas
    P      : pares (i, j), i < j, con partidos de alta audiencia
    S      : pares (i, j), i < j, que comparten estadio
    M_max  : dict {t: máx. partidos de alta audiencia} o entero para todas las jornadas
    w_b    : peso de breaks
    w_ext  : peso de visita en jornada crítica
    desactivar_cortes_heuristicas : desactiva cortes y heurísticas si es True
    Devuelve (modelo, calendario) con calendario = {t: [(local, visitante), ...]}.
    """
    # ------------------------------------------------------------------
    # CONJUNTOS
    # ------------------------------------------------------------------
    E = list(range(1, 2 * n + 1))
    J = list(range(1, 2 * n))
    E_EXT = list(E_EXT)
    J_EXT = list(J_EXT)
    P = [tuple(p) for p in P]
    S = [tuple(s) for s in S]
    if not isinstance(M_max, dict):
        M_max = {t: M_max for t in J}
    else:
        M_max = {int(t): val for t, val in M_max.items()}

    if not isinstance(n, int) or n < 1:
        raise ValueError("n debe ser un entero positivo.")
    if w_b < 0 or w_ext < 0:
        raise ValueError("Los pesos de la función objetivo no pueden ser negativos.")
    if not all(i in E for i in E_EXT):
        raise ValueError("E_EXT contiene equipos fuera de E.")
    if not all(t in J for t in J_EXT):
        raise ValueError("J_EXT contiene jornadas fuera de J.")
    if not all(i < j and i in E and j in E for i, j in P):
        raise ValueError("P contiene pares inválidos; cada par debe cumplir i < j y pertenecer a E.")
    if not all(i < j and i in E and j in E for i, j in S):
        raise ValueError("S contiene pares inválidos; cada par debe cumplir i < j y pertenecer a E.")
    if not all(t in M_max for t in J):
        raise ValueError("M_max debe estar definido para todas las jornadas.")
    if any(M_max[t] < 0 for t in J):
        raise ValueError("M_max no puede contener valores negativos.")
    if sum(M_max[t] for t in J) < len(P):
        raise ValueError("M_max es insuficiente para programar todos los pares de P.")

    # ------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------
    m = gp.Model("round_robin")
    if desactivar_cortes_heuristicas:
        m.setParam(GRB.Param.Cuts, 0)
        m.setParam(GRB.Param.Heuristics, 0)
    if not verbose:
        m.Params.OutputFlag = 0

    # Variables de decisión
    x = m.addVars(((i, j, t) for i in E for j in E if i != j for t in J),
                  vtype=GRB.BINARY, name="x")            # i local vs j en t
    y = m.addVars(E, J, vtype=GRB.BINARY, name="y")      # i es local en t

    T_BREAK = J[1:]                                      # t = 2..2n-1
    bH = m.addVars(E, T_BREAK, lb=0.0, name="bH")        # break de local
    bA = m.addVars(E, T_BREAK, lb=0.0, name="bA")        # break de visita
    v = m.addVars(E_EXT, J_EXT, lb=0.0, name="v")        # visita en jornada crítica

    # (1) Función objetivo
    m.setObjective(
        w_b * gp.quicksum(bH[i, t] + bA[i, t] for i in E for t in T_BREAK)
        + w_ext * gp.quicksum(v[i, t] for i in E_EXT for t in J_EXT),
        GRB.MINIMIZE,
    )

    # (2) Un partido por equipo por jornada
    m.addConstrs(
        (gp.quicksum(x[i, j, t] + x[j, i, t] for j in E if j != i) == 1
         for i in E for t in J),
        name="un_partido_por_jornada",
    )

    # (3) Cada par se enfrenta exactamente una vez
    m.addConstrs(
        (gp.quicksum(x[i, j, t] + x[j, i, t] for t in J) == 1
         for i in E for j in E if i < j),
        name="enfrentamiento_unico",
    )

    # (4) Enlace entre x e y, asignación de rival
    m.addConstrs(
        (y[i, t] == gp.quicksum(x[i, j, t] for j in E if j != i)
         for i in E for t in J),
        name="enlace_y",
    )

    # (5) Balance de localías: n-1 <= partidos de local <= n
    m.addConstrs((gp.quicksum(y[i, t] for t in J) >= n - 1 for i in E), name="local_min")
    m.addConstrs((gp.quicksum(y[i, t] for t in J) <= n for i in E), name="local_max")

    # (6) Break de local
    m.addConstrs(
        (bH[i, t] >= y[i, t - 1] + y[i, t] - 1 for i in E for t in T_BREAK),
        name="break_local",
    )

    # (7) Break de visita
    m.addConstrs(
        (bA[i, t] >= 1 - y[i, t - 1] - y[i, t] for i in E for t in T_BREAK),
        name="break_visita",
    )

    # (8) No más de 2 locales ni 2 visitas seguidas: 1 <= suma de 3 jornadas <= 2
    T_TRIPLE = J[1:-1]                                   # t = 2..2n-2
    m.addConstrs(
        (y[i, t - 1] + y[i, t] + y[i, t + 1] >= 1 for i in E for t in T_TRIPLE),
        name="max_visitas_seguidas",
    )
    m.addConstrs(
        (y[i, t - 1] + y[i, t] + y[i, t + 1] <= 2 for i in E for t in T_TRIPLE),
        name="max_locales_seguidos",
    )

    # (9) Penalización por ser visitante en jornada crítica de otra liga
    m.addConstrs(
        (v[i, t] >= 1 - y[i, t] for i in E_EXT for t in J_EXT),
        name="visita_jornada_critica",
    )

    # (10) Límite de partidos de alta audiencia por jornada
    m.addConstrs(
        (gp.quicksum(x[i, j, t] + x[j, i, t] for (i, j) in P) <= M_max[t] for t in J),
        name="limite_tv",
    )

    # (11) Equipos que comparten estadio no son locales a la vez
    m.addConstrs(
        (y[i, t] + y[j, t] <= 1 for (i, j) in S for t in J),
        name="estadio_compartido",
    )

    # ------------------------------------------------------------------
    # RESOLVER
    # ------------------------------------------------------------------
    m.setParam("TimeLimit", time_limit)
    m.Params.MIPGap = mip_gap
    m.optimize()

    calendario = {}
    if m.SolCount > 0:
        for t in J:
            calendario[t] = [(i, j) for i in E for j in E
                             if i != j and x[i, j, t].X > 0.5]
        if verbose:
            print(f"\nObjetivo: {m.ObjVal:.0f} | "
                  f"Cota inferior de breaks sin restricciones extra: {2 * n - 2}")
            for t in J:
                print(f"Jornada {t}: "
                      + "  ".join(f"{i}(L) vs {j}(V)" for i, j in calendario[t]))
    elif m.Status == GRB.INFEASIBLE:
        if verbose:
            print("Modelo infactible: revisa P, S, M_max, E_EXT y J_EXT.")

    return m, calendario