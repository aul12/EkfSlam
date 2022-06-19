import matplotlib
import json

import matplotlib.pyplot as plt

j = json.load(open("result.json", "r"))


def add_marker(obj, fig, marker, color):
    plt.plot(obj[0][0], obj[0][1], marker, color=color)


for snapshot in j:
    fig = plt.figure()
    add_marker(snapshot["vehicle"]["state"], fig, "x", "black")
    add_marker(snapshot["vehicle"]["est"], fig, "x", "blue")
    for cone in snapshot["cones"]:
        add_marker(cone, fig, "o", "black")

    if "estimatedCones" in snapshot:
        for cone in snapshot["estimatedCones"]:
            add_marker(cone, fig, "o", "blue")

    plt.gca().axis("equal")
    plt.show()
