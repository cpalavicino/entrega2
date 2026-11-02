"""
MH-GA para Tournament Scheduling (Round Robin simple, 2n equipos, 2n-1 jornadas).

Modelo de referencia: MODELO_NUEVO (variables x_ijt, y_it, b^H_it, b^A_it, v_it
y restricciones (2)-(15)).

Representacion (cromosoma) = (pi, sigma, h)
    pi    : permutacion de los 2n equipos sobre las 2n posiciones del metodo del circulo
    sigma : permutacion de las 2n-1 jornadas canonicas (orden en que se juegan)
    h     : vector binario, un bit por par {i,j} con i<j (1 => juega de local el menor)

La decodificacion construye SIEMPRE un Round Robin simple valido, por lo que
(2), (3), (4), (12), (13), (14), (15) se cumplen por construccion.
Las restricciones (5), (8), (10), (11) se controlan con una funcion de
violaciones, que se penaliza en el fitness y se reduce con un operador de reparacion.

Los indices internos van de 0; la salida se entrega con indices de 1 como en el modelo.
"""
import csv
import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path


# ----------------------------------------------------------------------------
# Datos del problema
# ----------------------------------------------------------------------------
@dataclass
class Instance:
    n: int                      # 2n equipos
    E_ext: set                  # equipos con otra competencia (indices desde 1)
    J_ext: set                  # jornadas criticas (indices desde 1)
    P: set                      # pares (i,j), i<j, de alta demanda televisiva
    S: set                      # pares (i,j), i<j, que comparten estadio
    Mmax: list                  # Mmax[t-1] para t = 1..2n-1
    w_b: float
    w_ext: float
    lam: float = 1000.0         # penalizacion por unidad de violacion

    def __post_init__(self):
        self.T = 2 * self.n                 # numero de equipos
        self.R = self.T - 1                 # numero de jornadas
        # versiones con indices desde 0
        self.E0 = set(range(self.T))
        self.Eext0 = {i - 1 for i in self.E_ext}
        self.Jext0 = {t - 1 for t in self.J_ext}
        self.P0 = {(i - 1, j - 1) for i, j in self.P}
        self.S0 = [(i - 1, j - 1) for i, j in self.S]
        self.pair_idx = {p: k for k, p in enumerate(combinations(range(self.T), 2))}
        self.M = len(self.pair_idx)         # numero de partidos = n(2n-1)
        self.canon = circle_method(self.T)  # calendario canonico


def circle_method(m):
    """Metodo del circulo: rounds[r] = lista de pares de posiciones (a,b)."""
    rounds = []
    for r in range(m - 1):
        pairs = [(m - 1, r)]
        for k in range(1, m // 2):
            pairs.append(((r + k) % (m - 1), (r - k) % (m - 1)))
        rounds.append(pairs)
    return rounds


# ----------------------------------------------------------------------------
# Cromosoma
# ----------------------------------------------------------------------------
@dataclass
class Chromosome:
    pi: list
    sigma: list
    h: list
    fit: float = field(default=float("inf"))
    z: float = field(default=float("inf"))
    V: int = field(default=10 ** 9)

    def copy(self):
        return Chromosome(self.pi[:], self.sigma[:], self.h[:], self.fit, self.z, self.V)


# ----------------------------------------------------------------------------
# Decodificacion: (pi, sigma, h) -> calendario
# ----------------------------------------------------------------------------
def decode(inst, ch):
    """
    Entrada : instancia y cromosoma.
    Salida  : games[t] = lista de (local, visita, idx_par) de la jornada t,
              y[i][t] = 1 si i es local en t, cntP[t] = partidos de P en t.
    """
    T, R = inst.T, inst.R
    games = []
    y = [[0] * R for _ in range(T)]
    cntP = [0] * R
    for t in range(R):
        r = ch.sigma[t]
        row = []
        for a, b in inst.canon[r]:
            i, j = ch.pi[a], ch.pi[b]
            lo, hi = (i, j) if i < j else (j, i)
            k = inst.pair_idx[(lo, hi)]
            home, away = (lo, hi) if ch.h[k] == 1 else (hi, lo)
            row.append((home, away, k))
            y[home][t] = 1
            if (lo, hi) in inst.P0:
                cntP[t] += 1
        games.append(row)
    return games, y, cntP


# ----------------------------------------------------------------------------
# Evaluacion: objetivo (1) y violaciones de (5), (8), (10), (11)
# ----------------------------------------------------------------------------
def objective(inst, y):
    """z = w_b * (breaks de local + breaks de visita) + w_ext * sum v_it."""
    breaks = 0
    for i in range(inst.T):
        yi = y[i]
        for t in range(1, inst.R):
            if yi[t - 1] == yi[t]:      # HH -> b^H = 1 ; AA -> b^A = 1
                breaks += 1
    v = sum(1 - y[i][t] for i in inst.Eext0 for t in inst.Jext0)
    return inst.w_b * breaks + inst.w_ext * v


def violations(inst, y, cntP, want_spots=False):
    """
    V = suma de violaciones de las restricciones no garantizadas por construccion.
    Si want_spots, devuelve ademas los pares (equipo, jornada) donde ocurren.
    """
    n, T, R = inst.n, inst.T, inst.R
    V = 0
    spots = []
    for i in range(T):                                  # (5)
        H = sum(y[i])
        if H < n - 1 or H > n:
            V += (n - 1 - H) if H < n - 1 else (H - n)
            if want_spots:
                spots.extend((i, t) for t in range(R))
        yi = y[i]
        for t in range(1, R - 1):                       # (8)
            s = yi[t - 1] + yi[t] + yi[t + 1]
            if s == 0 or s == 3:
                V += 1
                if want_spots:
                    spots.extend((i, u) for u in (t - 1, t, t + 1))
    for t in range(R):                                  # (10)
        if cntP[t] > inst.Mmax[t]:
            V += cntP[t] - inst.Mmax[t]
    for (i, j) in inst.S0:                              # (11)
        for t in range(R):
            if y[i][t] + y[j][t] > 1:
                V += 1
                if want_spots:
                    spots.append((i, t))
                    spots.append((j, t))
    return (V, spots) if want_spots else V


def evaluate(inst, ch):
    """fitness = z + lam * V  (se minimiza). Actualiza ch.z, ch.V, ch.fit."""
    _, y, cntP = decode(inst, ch)
    ch.z = objective(inst, y)
    ch.V = violations(inst, y, cntP)
    ch.fit = ch.z + inst.lam * ch.V
    return ch.fit


def is_feasible(inst, ch):
    _, y, cntP = decode(inst, ch)
    return violations(inst, y, cntP) == 0


# ----------------------------------------------------------------------------
# Busqueda local / reparacion (memetico)
# ----------------------------------------------------------------------------
def local_search(inst, ch, rng, flip_steps=60, swap_tries=6):
    """
    Fase 1 (local_flips): voltea el sentido local/visita de partidos.
    Fase 2: intercambia dos jornadas de sigma, reoptimiza h con pocas pasadas y
            acepta el cambio solo si el fitness mejora.
    Deja en ch los valores z, V y fit actualizados.
    """
    local_flips(inst, ch, rng, flip_steps)
    for _ in range(swap_tries):
        a, b = rng.sample(range(inst.R), 2)
        trial = ch.copy()
        trial.sigma[a], trial.sigma[b] = trial.sigma[b], trial.sigma[a]
        local_flips(inst, trial, rng, flip_steps // 2)
        if trial.fit < ch.fit:
            ch.pi, ch.sigma, ch.h = trial.pi, trial.sigma, trial.h
            ch.z, ch.V, ch.fit = trial.z, trial.V, trial.fit
    return ch


def local_flips(inst, ch, rng, steps):
    """
    Voltea el sentido de un partido, tomado de las violaciones con prob. 1/2 mientras
    V > 0 o al azar; acepta si el fitness (z + lam*V) no empeora. Actualiza ch.z/V/fit.
    """
    games, y, cntP = decode(inst, ch)
    V, spots = violations(inst, y, cntP, want_spots=True)
    z = objective(inst, y)
    fit = z + inst.lam * V
    gameAt = [{} for _ in range(inst.R)]
    for t in range(inst.R):
        for g in games[t]:
            gameAt[t][g[0]] = g
            gameAt[t][g[1]] = g
    for _ in range(steps):
        if V > 0 and spots and rng.random() < 0.5:
            i, t = spots[rng.randrange(len(spots))]
        else:
            i, t = rng.randrange(inst.T), rng.randrange(inst.R)
        home, away, k = gameAt[t][i]
        y[home][t], y[away][t] = 0, 1
        Vn, spn = violations(inst, y, cntP, want_spots=True)
        zn = objective(inst, y)
        fn = zn + inst.lam * Vn
        if fn <= fit:
            ch.h[k] ^= 1
            ng = (away, home, k)
            gameAt[t][home] = ng
            gameAt[t][away] = ng
            V, spots, z, fit = Vn, spn, zn, fn
        else:
            y[home][t], y[away][t] = 1, 0
    ch.z, ch.V, ch.fit = z, V, fit


# ----------------------------------------------------------------------------
# Poblacion inicial
# ----------------------------------------------------------------------------
def random_chromosome(inst, rng):
    pi = list(range(inst.T)); rng.shuffle(pi)
    sigma = list(range(inst.R)); rng.shuffle(sigma)
    h = [rng.randint(0, 1) for _ in range(inst.M)]
    return Chromosome(pi, sigma, h)


def initial_population(inst, N, rng, repair_steps, swap_tries):
    pop = []
    for _ in range(N):
        ch = random_chromosome(inst, rng)
        local_search(inst, ch, rng, repair_steps, swap_tries)
        pop.append(ch)
    return pop


# ----------------------------------------------------------------------------
# Operadores
# ----------------------------------------------------------------------------
def seleccion(pop, k, rng):
    """Seleccion por torneo de tamano k (minimiza fitness)."""
    return min(rng.sample(pop, k), key=lambda c: c.fit)


def order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n + 1), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    seg = set(p1[a:b])
    fill = [g for g in p2[b:] + p2[:b] if g not in seg]
    for pos, g in zip(list(range(b, n)) + list(range(a)), fill):
        child[pos] = g
    return child


def cut_crossover(h1, h2, rng):
    c = rng.randint(1, len(h1) - 1)
    return h1[:c] + h2[c:], h2[:c] + h1[c:]


def cruzamiento(p1, p2, pc, rng):
    """Con prob. pc: OX en pi y sigma, corte en h. Devuelve UN hijo."""
    child = p1.copy()
    if rng.random() < pc:
        child.pi = order_crossover(p1.pi, p2.pi, rng)
        child.sigma = order_crossover(p1.sigma, p2.sigma, rng)
        child.h = cut_crossover(p1.h, p2.h, rng)[0]
    return child


def inversion(perm, rng):
    a, b = sorted(rng.sample(range(len(perm) + 1), 2))
    perm[a:b] = perm[a:b][::-1]


def mutacion(ch, pm, rng):
    """Cada componente muta con prob. pm: inversion en pi, inversion en sigma,
    y bitwise en h (cada bit se voltea con prob. 2/|h|)."""
    if rng.random() < pm:
        inversion(ch.pi, rng)
    if rng.random() < pm:
        inversion(ch.sigma, rng)
    if rng.random() < pm:
        pb = 2.0 / len(ch.h)
        for k in range(len(ch.h)):
            if rng.random() < pb:
                ch.h[k] ^= 1


# ----------------------------------------------------------------------------
# Reemplazamiento
# ----------------------------------------------------------------------------
def reemplazamiento(P, P_new, elite):
    """Generacional con elitismo: los `elite` mejores de P + los mejores hijos."""
    P = sorted(P, key=lambda c: c.fit)
    P_new = sorted(P_new, key=lambda c: c.fit)
    return [c.copy() for c in P[:elite]] + P_new[:len(P) - elite]


# ----------------------------------------------------------------------------
# Algoritmo principal (misma estructura que el esquema visto en clase)
# ----------------------------------------------------------------------------
def genetic_algorithm(inst, N=50, pc=0.9, pm=0.3, G=500, k=3, elite=2,
                      repair_steps=60, swap_tries=6, seed=0, verbose=False,
                      time_limit=None):
    rng = random.Random(seed)
    t0 = time.time()
    P = initial_population(inst, N, rng, repair_steps, swap_tries)   # solucion_inicial + evaluar
    best = min(P, key=lambda c: c.fit).copy()
    history = []
    for g in range(G):                                               # criterio de termino
        P_new = []
        for t in range(N):
            p1 = seleccion(P, k, rng)
            p2 = seleccion(P, k, rng)
            c = cruzamiento(p1, p2, pc, rng)
            mutacion(c, pm, rng)
            local_search(inst, c, rng, repair_steps, swap_tries)     # mejora local (memetico)
            evaluate(inst, c)                                        # evaluar_individuo
            P_new.append(c)
        P = reemplazamiento(P, P_new, elite)
        cand = min(P, key=lambda c: c.fit)
        if cand.fit < best.fit:
            best = cand.copy()
        history.append(best.fit)
        if verbose and (g % 10 == 0 or g == G - 1):
            print(f"gen {g:4d}  best fit={best.fit:9.1f}  z={best.z:6.1f}  V={best.V}")
        if time_limit and time.time() - t0 > time_limit:
            break
    return solution_from(inst, best), history, time.time() - t0


# ----------------------------------------------------------------------------
# Salida: representacion con las variables del modelo (indices desde 1)
# ----------------------------------------------------------------------------
def solution_from(inst, ch):
    games, y, cntP = decode(inst, ch)
    x = {(h + 1, a + 1, t + 1) for t in range(inst.R) for (h, a, _) in games[t]}
    return {
        "x": x,                                                   # x_ijt = 1
        "y": {(i + 1, t + 1): y[i][t] for i in range(inst.T) for t in range(inst.R)},
        "z": ch.z,
        "violations": ch.V,
        "feasible": ch.V == 0,
        "chromosome": ch,
    }


# ----------------------------------------------------------------------------
# Verificador independiente: restricciones (2)-(15) tal como estan en el modelo
# ----------------------------------------------------------------------------
def verify(inst, x):
    """
    x: conjunto de (i, j, t), indices desde 1, con x_ijt = 1.
    Devuelve (factible, objetivo, lista de errores) evaluando el modelo literalmente.
    """
    E = range(1, inst.T + 1)
    J = range(1, inst.R + 1)
    n = inst.n
    errs = []
    X = lambda i, j, t: 1 if (i, j, t) in x else 0
    y = {(i, t): sum(X(i, j, t) for j in E if j != i) for i in E for t in J}     # (4)
    for i in E:                                                                  # (2)
        for t in J:
            if sum(X(i, j, t) + X(j, i, t) for j in E if j != i) != 1:
                errs.append(f"(2) equipo {i} jornada {t}")
    for i in E:                                                                  # (3)
        for j in E:
            if i < j and sum(X(i, j, t) + X(j, i, t) for t in J) != 1:
                errs.append(f"(3) par {i},{j}")
    for i in E:                                                                  # (5)
        if not (n - 1 <= sum(y[i, t] for t in J) <= n):
            errs.append(f"(5) equipo {i}")
    bH = {(i, t): max(0, y[i, t - 1] + y[i, t] - 1) for i in E for t in range(2, inst.R + 1)}   # (6)
    bA = {(i, t): max(0, 1 - y[i, t - 1] - y[i, t]) for i in E for t in range(2, inst.R + 1)}   # (7)
    for i in E:                                                                  # (8)
        for t in range(2, inst.R):
            if not (1 <= y[i, t - 1] + y[i, t] + y[i, t + 1] <= 2):
                errs.append(f"(8) equipo {i} jornada {t}")
    v = {(i, t): max(0, 1 - y[i, t]) for i in inst.E_ext for t in inst.J_ext}    # (9)
    for t in J:                                                                  # (10)
        if sum(X(i, j, t) + X(j, i, t) for (i, j) in inst.P) > inst.Mmax[t - 1]:
            errs.append(f"(10) jornada {t}")
    for (i, j) in inst.S:                                                        # (11)
        for t in J:
            if y[i, t] + y[j, t] > 1:
                errs.append(f"(11) par {i},{j} jornada {t}")
    z = inst.w_b * (sum(bH.values()) + sum(bA.values())) + inst.w_ext * sum(v.values())  # (1)
    return (not errs), z, errs


# ----------------------------------------------------------------------------
# Instancias de prueba
# ----------------------------------------------------------------------------
def example_instance(T=8, seed=1, w_b=10, w_ext=4):
    """Instancia reproducible con 2n = T equipos."""
    rng = random.Random(seed)
    n = T // 2
    R = T - 1
    teams = list(range(1, T + 1))
    E_ext = set(rng.sample(teams, max(2, T // 3)))
    J_ext = set(rng.sample(range(1, R + 1), 2))
    all_pairs = list(combinations(teams, 2))
    P = set(rng.sample(all_pairs, max(3, T // 2)))
    S = set()
    while len(S) < max(1, T // 4):
        a, b = sorted(rng.sample(teams, 2))
        S.add((a, b))
    Mmax = [1] * R
    return Instance(n, E_ext, J_ext, P, S, Mmax, w_b, w_ext)


def load_instance(csv_path, w_b=10, w_ext=4, critical_factor=0.40,
                  critical_seed=42):
    """Construye una instancia del algoritmo genético a partir de un CSV del proyecto."""
    csv_path = Path(csv_path)
    required_columns = {
        "Equipo",
        "Juega_Otras_Ligas",
        "ID_Estadio",
        "Peso_Audiencia",
    }
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path.name}: faltan columnas requeridas: "
                f"{', '.join(sorted(missing))}."
            )
        rows = list(reader)

    if not rows or len(rows) % 2:
        raise ValueError(f"{csv_path.name}: debe tener un número par de equipos.")

    teams = [row["Equipo"].strip() for row in rows]
    if any(not team for team in teams) or len(set(teams)) != len(teams):
        raise ValueError(f"{csv_path.name}: Equipo debe contener nombres únicos y no vacíos.")
    if any(not row["ID_Estadio"].strip() for row in rows):
        raise ValueError(f"{csv_path.name}: ID_Estadio no puede estar vacío.")
    if not 0 <= critical_factor <= 1:
        raise ValueError("critical_factor debe estar entre 0 y 1.")

    team_id = {team: i for i, team in enumerate(teams, start=1)}
    external_teams = []
    high_audience_teams = []
    teams_by_stadium = {}
    for row in rows:
        team = row["Equipo"].strip()
        try:
            plays_other_leagues = float(row["Juega_Otras_Ligas"])
            audience_weight = float(row["Peso_Audiencia"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{csv_path.name}: Juega_Otras_Ligas y Peso_Audiencia "
                "deben ser valores numéricos."
            ) from error
        if plays_other_leagues not in (0, 1):
            raise ValueError(
                f"{csv_path.name}: Juega_Otras_Ligas solo puede contener 0 o 1."
            )
        if plays_other_leagues == 1:
            external_teams.append(team_id[team])
        if audience_weight == 100:
            high_audience_teams.append(team_id[team])
        stadium = row["ID_Estadio"].strip()
        teams_by_stadium.setdefault(stadium, []).append(team_id[team])

    total_teams = len(teams)
    rounds = total_teams - 1
    critical_count = math.ceil(
        rounds * critical_factor * len(external_teams) / total_teams
    )
    instance_seed = hashlib.sha256(
        f"{critical_seed}:{csv_path.name}".encode("utf-8")
    ).digest()
    critical_rng = random.Random(int.from_bytes(instance_seed[:8], "big"))
    critical_rounds = set(
        critical_rng.sample(range(1, rounds + 1), critical_count)
    )

    high_audience_pairs = set(combinations(sorted(high_audience_teams), 2))
    shared_stadium_pairs = {
        pair
        for stadium_teams in teams_by_stadium.values()
        for pair in combinations(sorted(stadium_teams), 2)
    }
    return Instance(
        n=total_teams // 2,
        E_ext=set(external_teams),
        J_ext=critical_rounds,
        P=high_audience_pairs,
        S=shared_stadium_pairs,
        Mmax=[len(high_audience_teams) // 2] * rounds,
        w_b=w_b,
        w_ext=w_ext,
    )


def find_instances_dir():
    """Busca la carpeta INSTANCIAS junto al script, al directorio actual o al proyecto."""
    roots = (Path(__file__).resolve().parent, Path.cwd())
    candidates = []
    for root in roots:
        candidates.extend((root / "INSTANCIAS", root / "entrega2" / "INSTANCIAS"))

    desktop = Path.home() / "OneDrive" / "Escritorio"
    if desktop.is_dir():
        candidates.extend(
            project / "entrega2" / "INSTANCIAS"
            for project in desktop.iterdir()
            if project.is_dir()
        )

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
    raise FileNotFoundError(
        "No se encontró una carpeta INSTANCIAS con archivos CSV. "
        "Colócala junto al script o ejecuta el script desde la carpeta del proyecto."
    )


def mutation_rate_for_size(team_count, min_teams=8, max_teams=16,
                           max_rate=0.3, min_rate=0.1):
    """Interpola linealmente la mutación entre 8 y 16 equipos."""
    if min_teams >= max_teams:
        raise ValueError("max_teams debe ser mayor que min_teams.")
    if not 0 <= min_rate <= max_rate <= 1:
        raise ValueError("Las tasas deben cumplir 0 <= min_rate <= max_rate <= 1.")
    progress = (team_count - min_teams) / (max_teams - min_teams)
    progress = min(1.0, max(0.0, progress))
    return max_rate + progress * (min_rate - max_rate)


def run_all_instances(instances_dir=None, w_b=10, w_ext=4, verbose=False,
                      seeds=range(1, 6), generations=500):
    """Ejecuta cada CSV para las semillas indicadas y guarda un reporte."""
    instances_dir = Path(instances_dir) if instances_dir else find_instances_dir()
    instance_files = sorted(instances_dir.glob("*.csv"))
    if not instance_files:
        raise FileNotFoundError(f"No hay archivos CSV en {instances_dir}.")
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("Debe indicarse al menos una semilla.")
    if generations < 1:
        raise ValueError("generations debe ser un entero positivo.")

    results = []
    for csv_path in instance_files:
        instance = load_instance(csv_path, w_b=w_b, w_ext=w_ext)
        population_size = math.ceil(50 * instance.T / 8)
        mutation_rate = mutation_rate_for_size(instance.T)
        for seed in seeds:
            random.seed(seed)
            solution, history, elapsed = genetic_algorithm(
                instance,
                N=population_size,
                pm=mutation_rate,
                G=generations,
                seed=seed,
                verbose=verbose,
            )
            feasible, verified_objective, errors = verify(instance, solution["x"])
            result = {
                "Instancia": csv_path.name,
                "Semilla": seed,
                "Equipos": instance.T,
                "Poblacion": population_size,
                "Probabilidad_mutacion": mutation_rate,
                "Jornadas_criticas": sorted(instance.J_ext),
                "Fitness": solution["z"] + instance.lam * solution["violations"],
                "Objetivo": solution["z"],
                "Violaciones": solution["violations"],
                "Factible": solution["feasible"],
                "Verificacion_independiente": feasible,
                "Objetivo_verificado": verified_objective,
                "Tiempo_s": elapsed,
                "Generaciones_completadas": len(history),
                "Errores_verificacion": errors,
            }
            results.append(result)
            print(
                f"{csv_path.name} | semilla={seed} | población={population_size} "
                f"| mutación={mutation_rate:.3f} "
                f"| objetivo={solution['z']:.2f} | violaciones={solution['violations']} "
                f"| factible={feasible} | tiempo={elapsed:.1f}s"
            )

    output_dir = instances_dir.parent / "RESULTADOS"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "Resultados_GA_5_semillas.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Reporte guardado en: {report_path}")
    return results


if __name__ == "__main__":
    run_all_instances()
