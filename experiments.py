"""Checks the numbers used in the report's analysis.

Run in the same folder as ga.py and program_ratings_modified.csv:
    python experiments.py
"""
from ga import genetic_algorithm, optimal_fitness, read_csv_to_dict

ratings = read_csv_to_dict("program_ratings_modified.csv")
BEST = optimal_fitness(ratings)  # 9.7


def first_reached(history):
    """Generation where the best fitness first reached the best possible total."""
    for gen, fit in enumerate(history):
        if fit >= BEST - 1e-6:
            return gen
    return None


settings = [(0.60, 0.05), (0.12, 0.03), (0.95, 0.01), (0.80, 0.20)]

print("Single runs (seed 1, same as the app)")
for co_r, mut_r in settings:
    _, fit, hist = genetic_algorithm(ratings, crossover_rate=co_r,
                                     mutation_rate=mut_r, seed=1)
    g = first_reached(hist)
    print(f"  CO_R {co_r:.2f}, MUT_R {mut_r:.2f}: total {fit:.1f}, "
          f"9.7 reached at {'generation ' + str(g) if g is not None else 'never'}, "
          f"fitness at generation 20 = {hist[20]:.1f}")

print("\n10 runs each (seeds 1 to 10)")
for co_r, mut_r in settings + [(0.60, 0.03)]:
    fits, gens = [], []
    for seed in range(1, 11):
        _, fit, hist = genetic_algorithm(ratings, crossover_rate=co_r,
                                         mutation_rate=mut_r, seed=seed)
        fits.append(fit)
        g = first_reached(hist)
        if g is not None:
            gens.append(g)
    avg_gen = f"{sum(gens) / len(gens):.1f}" if gens else "not reached"
    print(f"  CO_R {co_r:.2f}, MUT_R {mut_r:.2f}: average total {sum(fits) / 10:.2f}, "
          f"reached 9.7 in {len(gens)} of 10 runs, average generations {avg_gen}")
