import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ==================================================
# PARAMETERS
# ==================================================
POP_SIZE = 25
GENOME_LENGTH = 6
GENERATIONS = 100
MUTATION_RATE = 0.1

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
def simulate_robot(genome, steps=200, start_pos=None):
    # === INIT ===
    if start_pos is None:
        x, y = np.random.uniform(-5, 5, 2)
    else:
        x, y = start_pos

    theta = np.random.uniform(-np.pi, np.pi)

    # Step 1 constants
    k_speed = 2.0
    RADIUS = 0.1
    b = np.pi / 4

    path = [(x, y)]

    for _ in range(steps):

        # === DISTANCE TO LIGHT ===
        d2 = (LIGHT_POS[0] - x)**2 + (LIGHT_POS[1] - y)**2

        # === SENSOR POSITIONS ===
        lsx = x + np.cos(theta + b) * RADIUS
        lsy = y + np.sin(theta + b) * RADIUS

        rsx = x + np.cos(theta - b) * RADIUS
        rsy = y + np.sin(theta - b) * RADIUS

        # === SENSOR DISTANCE ===
        dl2 = (LIGHT_POS[0] - lsx)**2 + (LIGHT_POS[1] - lsy)**2
        dr2 = (LIGHT_POS[0] - rsx)**2 + (LIGHT_POS[1] - rsy)**2

        # === OCCLUSION ===
        if dl2 > d2:
            ls_stimulation = 0.0
        else:
            ls_stimulation = 1.0 / (dl2 + 0.01)

        if dr2 > d2:
            rs_stimulation = 0.0
        else:
            rs_stimulation = 1.0 / (dr2 + 0.01)

        # clamp sensors
        ls_stimulation = min(ls_stimulation, 5.0)
        rs_stimulation = min(rs_stimulation, 5.0)

        # === CONTROLLER (GA GENOME) ===
        L = genome[0] + genome[1]*rs_stimulation + genome[2]*ls_stimulation
        R = genome[3] + genome[4]*rs_stimulation + genome[5]*ls_stimulation

        L = np.clip(L, -2.0, 5.0)
        R = np.clip(R, -2.0, 5.0)

        # === STEP 1 MOVEMENT (IMPORTANT) ===
        dxdt = (L + R) * np.cos(theta) * k_speed
        dydt = (L + R) * np.sin(theta) * k_speed
        dodt = (R - L) * 8.0

        theta += dodt * 0.02
        x += dxdt * 0.02
        y += dydt * 0.02

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

    total = 0

    for start in start_positions:
        path = simulate_robot(genome, start_pos=start)

        distances = np.sqrt(
            (path[:,0] - LIGHT_POS[0])**2 +
            (path[:,1] - LIGHT_POS[1])**2
        )

        final_d = distances[-1]

        # reward closeness
        total += 1 / (final_d + 0.1)

        # strong reward for reaching light
        if final_d < 0.5:
            total += 20

        # reward improvement (prevents circling)
        total += (distances[0] - final_d)

    return total / len(start_positions)


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

        if np.random.rand() < 0.75:
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
        else:
            winner = population[np.random.randint(POP_SIZE)]

        child = winner.copy()

        # mutation
        for k in range(GENOME_LENGTH):
            if np.random.rand() < MUTATION_RATE:
                child[k] += np.random.randn() * 0.1

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
    plt.scatter(start[0], start[1], c="black", s=20)
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