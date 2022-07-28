import json

import matplotlib.pyplot as plt

j = json.load(open("result.json", "r"))


def add_marker(obj, marker, color, fillstyle="full"):
    plt.plot(obj[0][0], obj[0][1], marker, color=color, fillstyle=fillstyle)


c = 0
for snapshot in j:
    fig = plt.figure()
    add_marker(snapshot["vehicle"]["state"], "x", "black")
    add_marker(snapshot["vehicle"]["est"], "x", "blue")
    for cone in snapshot["cones"]:
        add_marker(cone["state"], "o", cone["color"], fillstyle="none")

    if "estimatedCones" in snapshot:
        for cone in snapshot["estimatedCones"]:
            add_marker(cone["state"], "o", cone["color"])

    plt.gca().axis("equal")
    plt.savefig(f"out/{c}.png")
    c += 1
    print(f"\rSaving {c}", end="")
    plt.close(fig)
