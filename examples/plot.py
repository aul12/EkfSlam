import json
import math

import matplotlib.pyplot as plt

j = json.load(open("result.json", "r"))


def add_cone(obj, color, fillstyle):
    plt.plot(obj[0][0], obj[0][1], marker="o", color=color, fillstyle=fillstyle)


def add_vehicle(obj, color):
    x_pos = obj[0][0]
    y_pos = obj[0][1]
    yaw = obj[0][3]

    width = 2.5
    height = 1.5
    angle_factor = 0.7
    width_vec = [math.cos(yaw) * width / 2, math.sin(yaw) * width / 2]
    height_vec = [-math.sin(yaw) * height / 2, math.cos(yaw) * height / 2]
    points = [
        [x_pos - width_vec[0] - height_vec[0], y_pos - width_vec[1] - height_vec[1]],  # BL
        [x_pos + width_vec[0] * angle_factor - height_vec[0], y_pos + width_vec[1] * angle_factor - height_vec[1]],
        # BR
        [x_pos + width_vec[0], y_pos + width_vec[1]],  # front
        [x_pos + width_vec[0] * angle_factor + height_vec[0], y_pos + width_vec[1] * angle_factor + height_vec[1]],
        # TR
        [x_pos - width_vec[0] + height_vec[0], y_pos - width_vec[1] + height_vec[1]],  # TL
    ]

    plt.gca().add_patch(plt.Polygon(points, closed=True, alpha=.5, color=color))


c = 0
for snapshot in j:
    fig = plt.figure()
    add_vehicle(snapshot["vehicle"]["state"], color="green")
    add_vehicle(snapshot["vehicle"]["est"], color="red")
    for cone in snapshot["cones"]:
        add_cone(cone["state"], cone["color"], fillstyle="none")

    if "estimatedCones" in snapshot:
        for cone in snapshot["estimatedCones"]:
            add_cone(cone["state"], cone["color"], fillstyle="full")

    plt.gca().axis("equal")
    plt.savefig(f"out/{c}.svg")
    c += 1
    print(f"\rSaving {c}", end="")
    plt.close(fig)
