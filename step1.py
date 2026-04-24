import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# GLOBALS
# ==================================================
k_speed = 2.0
RADIUS = 0.1
b = np.pi / 4

light_x = 0.0
light_y = 0.0


# ==================================================
# ROBOT DERIVATIVE
# ==================================================
def robot_derivative(state, mode):
    x, y, o = state

    # distance robot -> light
    d2 = (light_x - x)**2 + (light_y - y)**2

    # sensor positions
    lsx = x + np.cos(o + b) * RADIUS
    lsy = y + np.sin(o + b) * RADIUS

    rsx = x + np.cos(o - b) * RADIUS
    rsy = y + np.sin(o - b) * RADIUS

    # sensor -> light distance
    dl2 = (light_x - lsx)**2 + (light_y - lsy)**2
    dr2 = (light_x - rsx)**2 + (light_y - rsy)**2

    # occlusion
    ls_is_obscured = dl2 > d2
    rs_is_obscured = dr2 > d2

    # stimulation
    ls_stimulation = 1.0 / (dl2 + 0.01)
    rs_stimulation = 1.0 / (dr2 + 0.01)

    if ls_is_obscured:
        ls_stimulation = 0.0
    if rs_is_obscured:
        rs_stimulation = 0.0

    # keep values in a reasonable range
    ls_stimulation = min(ls_stimulation, 5.0)
    rs_stimulation = min(rs_stimulation, 5.0)

    # ==================================================
    # BEHAVIOURS
    # ==================================================
    if mode == "love":
        L = rs_stimulation
        R = ls_stimulation

    elif mode == "aggression":
        L = ls_stimulation
        R = rs_stimulation

    elif mode == "explorer":
        L = 1.5 - rs_stimulation
        R = 1.5 - ls_stimulation

    elif mode == "fear":
        L = 1.5 - ls_stimulation
        R = 1.5 - rs_stimulation

    else:
        raise ValueError("mode must be love, aggression, explorer, or fear")

    # stop motors going too negative or too huge
    L += 0.05
    R -= 0.05
    L = np.clip(L, -2.0, 5.0)
    R = np.clip(R, -2.0, 5.0)

    # motion
    dxdt = (L + R) * np.cos(o) * k_speed
    dydt = (L + R) * np.sin(o) * k_speed
    dodt = (R - L) * 8.0

    return [dxdt, dydt, dodt]


# ==================================================
# TRAJECTORY
# ==================================================
def trajectory(dur, init_con, mode):
    DT = 0.02
    N_ITS = int(dur / DT)

    x, y, o = init_con

    xs = [x]
    ys = [y]
    os = [o]

    for _ in range(N_ITS):
        dxdt, dydt, dodt = robot_derivative([x, y, o], mode)

        x = x + dxdt * DT
        y = y + dydt * DT
        o = o + dodt * DT

        xs.append(x)
        ys.append(y)
        os.append(o)

    return xs, ys, os


# ==================================================
# RUN + PLOT
# ==================================================
if __name__ == "__main__":
    modes = ["love", "aggression", "explorer", "fear"]

    for mode in modes:
        plt.figure(figsize=(7, 7))

        for x in np.linspace(-5, 5, 5):
            for y in np.linspace(-5, 5, 5):
                init_con = [x, y, np.pi / 2]
                xs, ys, _ = trajectory(10, init_con, mode)

                plt.plot(xs, ys)
                plt.plot(xs[0], ys[0], "go", markersize=4)
                plt.plot(xs[-1], ys[-1], "rx", markersize=5)

        plt.scatter([light_x], [light_y], color="red", s=120, label="light")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(mode.capitalize())
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()