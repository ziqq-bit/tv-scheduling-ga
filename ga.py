"""Genetic algorithm for TV programme scheduling (same method as the Colab notebook)."""
import csv
import random

ALL_TIME_SLOTS = list(range(6, 24))  # 06:00 to 23:00, one programme per hour


def read_csv_to_dict(file_path):
    """Read the ratings CSV into {programme: [rating per hour]}."""
    program_ratings = {}
    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            if row:
                program_ratings[row[0]] = [float(x) for x in row[1:]]
    return program_ratings


def calculate_fitness(chromosome, ratings):
    """Total rating of a schedule: sum of each programme's rating in its hour."""
    return sum(ratings[program][hour] for hour, program in enumerate(chromosome))


def initialize_pop(programs, n_slots, pop_size):
    """Random initial population: each chromosome picks one programme per hour."""
    return [random.choices(programs, k=n_slots) for _ in range(pop_size)]


def selection(population, ratings, k=3):
    """Tournament selection: best of 3 random chromosomes, repeated POP_SIZE times."""
    return [max(random.sample(population, k), key=lambda c: calculate_fitness(c, ratings))
            for _ in range(len(population))]


def crossover(parent1, parent2):
    """One-point crossover."""
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def mutation(chromosome, programs, mut_r):
    """Per-gene mutation: each hour has probability MUT_R of a random new programme."""
    chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < mut_r:
            chromosome[i] = random.choice(programs)
    return chromosome


def genetic_algorithm(ratings, generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, seed=None):
    """Run the GA. Returns (best_schedule, best_fitness, best_fitness_history)."""
    if seed is not None:
        random.seed(seed)
    programs = list(ratings.keys())
    n_slots = len(next(iter(ratings.values())))
    population = initialize_pop(programs, n_slots, population_size)
    history = []

    for _ in range(generations):
        best = max(population, key=lambda c: calculate_fitness(c, ratings))
        history.append(calculate_fitness(best, ratings))
        selected = selection(population, ratings)

        new_population = [best.copy()]  # elitism: keep the best schedule
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(selected), random.choice(selected)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            new_population.append(mutation(child1, programs, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutation(child2, programs, mutation_rate))
        population = new_population

    best = max(population, key=lambda c: calculate_fitness(c, ratings))
    history.append(calculate_fitness(best, ratings))
    return best, calculate_fitness(best, ratings), history


def optimal_fitness(ratings):
    """Upper bound: highest-rated programme in every hour."""
    n_slots = len(next(iter(ratings.values())))
    return sum(max(r[i] for r in ratings.values()) for i in range(n_slots))
