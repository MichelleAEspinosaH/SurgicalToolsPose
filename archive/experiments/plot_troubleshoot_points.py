import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection


def main() -> None:
    # points = [
    #     (0.1, 1.1, 4.0),
    #     (-1.0, 1.0, 3.2),
    #     (1.4, -1.1, 4.4),
    #     (1.5, 0.5, 3.5),
    #     (0.2, 0.4, 2.8),
    #     (-0.1, 1.1, 6.9),
    # ]
    # points = [
    #     (0.1, 1.3, 4.7),
    #     (-1.9,-2.2,4.9),
    #     (3.6,-2.6,4.9),
    #     (5.2,0.2,7.6),
    #     (1.1,0.7,1.9),
    #     (0.1,0.4,1.7),
    #     (-0.9,0.4,1.3)
    # ]
    points = [
        (-0.3, 1.2, 3.6),
        (-1.5, 0.4, 2.1),
        (-3.2,-2.6, 4.7),
        (3.9, -2.9, 5.4),
        (3.4, 0.3, 5.2),
        (2.7, 1.7, 3.4)
    ]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    ax.scatter(xs, ys, zs, color="royalblue", s=60)

    for idx, (x, y, z) in enumerate(points, start=1):
        ax.text(x, y, z, f"{idx}", color="black", fontsize=11)

    ax.set_title("Troubleshooting Points (Labeled 1-6)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
