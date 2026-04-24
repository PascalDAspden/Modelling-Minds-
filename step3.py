import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ==================================================
# PARAMETERS
# ==================================================
POP_SIZE = 50
GENOME_LENGTH = 6
GENERATIONS = 3000
MUTATION_RATE = 0.05

LIGHT_POS = np.array([0.0, 0.0])

# ==================================================
# INITIALISE POPULATION
# 6 genes: p0, p1, p2, p3, p4, p5
# ==================================================
population = np.random.uniform(-1, 1, (POP_SIZE, GENOME_LENGTH))

fitness_history = []

early_pop = None
mid_pop = None
late_pop = None

# ==================================================
# ROBOT SIMULATION
# ==================================================
def simulate_robot(genome, steps=150, start_pos=None):
    if start_pos is None:
        x, y = np.random.uniform(-5, 5, 2)
    else:
        x, y = start_pos

    theta = np.random.uniform(-np.pi, np.pi)
    path = [(x, y)]

    for _ in range(steps):
        dx = LIGHT_POS[0] - x
        dy = LIGHT_POS[1] - y
        distance = np.sqrt(dx**2 + dy**2)

        # simple sensor strength based on distance to light
        sensor_strength = 1 / (distance + 0.1)

        # left and right sensors
        Sleft = sensor_strength
        Sright = sensor_strength

        # required assignment controller:
        # Mleft = p0 + p1*Sright + p2*Sleft
        # Mright = p3 + p4*Sright + p5*Sleft
        Mleft = genome[0] + genome[1] * Sright + genome[2] * Sleft
        Mright = genome[3] + genome[4] * Sright + genome[5] * Sleft

        # limit motor values
        Mleft = np.tanh(Mleft)
        Mright = np.tanh(Mright)

        # robot movement
        speed = (Mleft + Mright) / 2
        turn = (Mright - Mleft) / 2

        theta += turn

        x += speed * np.cos(theta)
        y += speed * np.sin(theta)

        path.append((x, y))

    return np.array(path)


# ==================================================
# FITNESS FUNCTION
# ==================================================
def fitness(genome):
    start_positions = [
        (-5, -5), (-5, 0), (-5, 5),
        (0, -5),          (0, 5),
        (5, -5),  (5, 0), (5, 5),
        (-3, -3), (-3, 3), (3, -3), (3, 3),
        (-5, 2), (5, 2), (-2, 5), (2, -5)
    ]

    total_fitness = 0

    for start in start_positions:
        path = simulate_robot(genome, start_pos=start)
        final_pos = path[-1]

        final_distance = np.sqrt(
            (final_pos[0] - LIGHT_POS[0])**2 +
            (final_pos[1] - LIGHT_POS[1])**2
        )

        # assignment fitness: negative distance squared
        total_fitness += -(final_distance ** 2)

    return total_fitness / len(start_positions)


# ==================================================
# EVOLUTION LOOP
# ==================================================
for gen in range(GENERATIONS):

    fitnesses = np.array([fitness(ind) for ind in population])
    fitness_history.append(fitnesses.copy())

    if gen == 0:
        early_pop = population.copy()

    if gen == GENERATIONS // 2:
        mid_pop = population.copy()

    if gen == GENERATIONS - 1:
        late_pop = population.copy()

    new_population = []

    for _ in range(POP_SIZE):
        i, j = np.random.randint(0, POP_SIZE, 2)

        if fitnesses[i] > fitnesses[j]:
            winner = population[i]
        else:
            winner = population[j]

        child = winner.copy()

        # mutation
        for k in range(GENOME_LENGTH):
            if np.random.rand() < MUTATION_RATE:
                child[k] += np.random.randn() * 0.02

        # clip genes between -1 and 1
        child = np.clip(child, -1, 1)

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
plt.title("GA Fitness for Evolved Robot")
plt.legend()
plt.grid()
plt.show()


# ==================================================
# PLOT BEHAVIOUR
# ==================================================
def plot_behaviour(pop, title):
    plt.figure()

    start_positions = [
        (-5, -5), (-5, 0), (-5, 5),
        (0, -5),          (0, 5),
        (5, -5),  (5, 0), (5, 5),
        (-3, -3), (-3, 3), (3, -3), (3, 3),
        (-5, 2), (5, 2), (-2, 5), (2, -5)
    ]

    # best individual from this population
    fitnesses = np.array([fitness(ind) for ind in pop])
    best_genome = pop[np.argmax(fitnesses)]

    for start in start_positions:
        path = simulate_robot(best_genome, start_pos=start)
        plt.plot(path[:, 0], path[:, 1])

    plt.scatter(LIGHT_POS[0], LIGHT_POS[1], c="red", s=100, label="Light")
    plt.title(title)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid()
    plt.show()


plot_behaviour(early_pop, "Early Best Robot Behaviour")
plot_behaviour(mid_pop, "Mid Best Robot Behaviour")
plot_behaviour(late_pop, "Late Best Robot Behaviour")