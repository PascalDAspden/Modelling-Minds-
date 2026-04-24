import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# PARAMETERS
# ==================================================
POP_SIZE = 50
GENOME_LENGTH = 10
GENERATIONS = 3000
MUTATION_RATE = 0.05

LIGHT_POS = np.array([0.0, 0.0])

# ==================================================
# INITIALISE POPULATION
# ==================================================
population = np.random.uniform(-1, 1, (POP_SIZE, GENOME_LENGTH))

fitness_history = []

# ==================================================
# ROBOT SIMULATION
# ==================================================
def simulate_robot(genome, steps=100):
    x, y = np.random.uniform(-5, 5, 2)
    path = [(x, y)]

    prev_d = None
    total = 0

    for _ in range(steps):
        dx, dy = LIGHT_POS - np.array([x, y])
        d = np.sqrt(dx**2 + dy**2)

        # movement
        vx = genome[0]*dx + genome[1]*dy + genome[2]
        vy = genome[3]*dx + genome[4]*dy + genome[5]

        x += np.tanh(vx)
        y += np.tanh(vy)

        path.append((x, y))

        # =========================
        # FITNESS (TUNED)
        # =========================

        # reward getting close
        total += 1 / (d + 0.1)

        # BIG reward for reaching light
        if d < 1.0:
            total += 5

        # penalise orbiting (not improving)
        if prev_d is not None and abs(prev_d - d) < 0.02:
            total -= 4

        prev_d = d

    return total, path

# ==================================================
# FITNESS FUNCTION
# ==================================================
def fitness(genome):
    score, _ = simulate_robot(genome)
    return score

# ==================================================
# EVOLUTION LOOP
# ==================================================
for gen in range(GENERATIONS):

    fitnesses = np.array([fitness(ind) for ind in population])
    fitness_history.append(fitnesses.copy())

    new_population = []

    for _ in range(POP_SIZE):
        i, j = np.random.randint(0, POP_SIZE, 2)

        winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
        child = winner.copy()

        # mutation
        for k in range(GENOME_LENGTH):
            if np.random.rand() < MUTATION_RATE:
                child[k] += np.random.normal(0, 0.3)

        new_population.append(child)

    population = np.array(new_population)

# ==================================================
# PLOT FITNESS
# ==================================================
max_fit = [np.max(f) for f in fitness_history]
mean_fit = [np.mean(f) for f in fitness_history]
min_fit = [np.min(f) for f in fitness_history]

plt.figure()
plt.plot(max_fit, label="max")
plt.plot(mean_fit, label="mean")
plt.plot(min_fit, label="min")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("GA Learning Robot Behaviour")
plt.legend()
plt.show()

# ==================================================
# PLOT BEHAVIOUR
# ==================================================
def plot_behaviour(pop, title):
    plt.figure()

    for i in range(20):
        _, path = simulate_robot(pop[i])
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1])

    plt.scatter(LIGHT_POS[0], LIGHT_POS[1], c='red', s=100)
    plt.title(title)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()

    plt.show()

# early / mid / late
plot_behaviour(population[:20], "Late Behaviour")

# re-run quick snapshots for early/mid
# (simple approximation)
early_pop = np.random.uniform(-1, 1, (POP_SIZE, GENOME_LENGTH))
mid_pop = population.copy()

plot_behaviour(early_pop, "Early Behaviour")
plot_behaviour(mid_pop, "Mid Behaviour")