import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
# ==================================================
# PARAMETERS
# ==================================================
N_GENES = 64
WORD_LENGTH = 4
N_WORDS = N_GENES // WORD_LENGTH

POP_SIZE = 100
MUTATION_RATE = 0.01

population = np.random.randint(0, 2, size=(POP_SIZE, N_GENES))
fitnesses = np.zeros(POP_SIZE)

fitnesses_h = []


# ==================================================
# PLOT POPULATION 
# ==================================================
def plot_population(pop, title):
    plt.figure()
    plt.imshow(pop, aspect='auto', cmap='gray')
    plt.title(title)
    plt.xlabel("Genes")
    plt.ylabel("Individuals")
    plt.colorbar()
    plt.show()


# ==================================================
# FITNESS FUNCTION
# ==================================================
def fitness(individual):
    f = 0
    for word_i in range(N_WORDS):
        word = individual[word_i * WORD_LENGTH:(word_i + 1) * WORD_LENGTH]
        if np.sum(word) == WORD_LENGTH:
            f += 1
    return f / N_WORDS


# ==================================================
# EVOLUTION LOOP
# ==================================================
trial = 0

while trial < 20000:
    trial += 1

    p1 = np.random.randint(POP_SIZE)
    p2 = p1

    while p2 == p1:
        p2 = np.random.randint(POP_SIZE)

    f1 = fitness(population[p1])
    f2 = fitness(population[p2])

    fitnesses[p1] = f1
    fitnesses[p2] = f2

    fitnesses_h.append(fitnesses.copy())

    if f1 > f2:
        winner = p1
        loser = p2
    else:
        winner = p2
        loser = p1

    for j in range(N_GENES):

        # mutation
        if np.random.rand() < MUTATION_RATE:
            population[loser][j] = 1 - population[loser][j]

        # crossover
        if np.random.rand() < 0.75:
            population[loser][j] = population[winner][j]

    # ==================================================
    # POPULATION SNAPSHOTS
    # ==================================================
    if trial == 100:
        plot_population(population, "Early Population")

    if trial == 5000:
        plot_population(population, "Mid Population")

    if trial == 20000:
        plot_population(population, "Late Population")


# ==================================================
# PLOT FITNESS
# ==================================================
plt.figure()

max_fits = [np.max(f) for f in fitnesses_h]
min_fits = [np.min(f) for f in fitnesses_h]
avg_fits = [np.mean(f) for f in fitnesses_h]

plt.plot(max_fits, label="max")
plt.plot(min_fits, label="min")
plt.plot(avg_fits, label="mean")

plt.legend()
plt.xlabel("Trial")
plt.ylabel("Fitness")
plt.title("GA Fitness (Royal Road)")

plt.show()