"""
Estimate and draw pose axes for three surgical tools in an image.
Axes convention: X=red (along tool length), Y=green (perpendicular in-plane), Z=blue (out-of-plane)
"""

import cv2
import numpy as np
import sys
import os

def draw_axes(img, center, angle_deg, axis_len=80, thickness=3):
    """
    Draw 3D-style pose axes on image.
    angle_deg: angle of tool's principal axis from +X (horizontal right), in degrees.
    X (red)  → along tool length (tip direction)
    Y (green)→ perpendicular to tool, in image plane
    Z (blue) → out of image plane (foreshortened at 45°, drawn up-right to simulate depth)
    """
    cx, cy = int(center[0]), int(center[1])
    theta = np.radians(angle_deg)

    # X axis — along tool
    x_end = (int(cx + axis_len * np.cos(theta)),
             int(cy + axis_len * np.sin(theta)))
    # Y axis — perpendicular (rotated 90°)
    y_end = (int(cx + axis_len * np.cos(theta + np.pi/2)),
             int(cy + axis_len * np.sin(theta + np.pi/2)))
    # Z axis — simulate out-of-plane with foreshortening (45° upper-right, half length)
    z_len = axis_len * 0.6
    z_end = (int(cx + z_len * np.cos(np.radians(-60))),
             int(cy + z_len * np.sin(np.radians(-60))))

    cv2.arrowedLine(img, (cx, cy), x_end, (0, 0, 220), thickness, tipLength=0.25)   # X red
    cv2.arrowedLine(img, (cx, cy), y_end, (0, 200, 0), thickness, tipLength=0.25)   # Y green
    cv2.arrowedLine(img, (cx, cy), z_end, (220, 80, 0), thickness, tipLength=0.25)  # Z blue

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "X", (x_end[0]+4, x_end[1]+4), font, 0.55, (0,0,220), 2)
    cv2.putText(img, "Y", (y_end[0]+4, y_end[1]+4), font, 0.55, (0,200,0), 2)
    cv2.putText(img, "Z", (z_end[0]+4, z_end[1]+4), font, 0.55, (220,80,0), 2)


def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # ── Estimated poses (visual inspection) ──────────────────────────────────
    # Each entry: (label, center_x_frac, center_y_frac, angle_deg)
    #   angle_deg = angle of tip direction from +X (image right), measured CW positive
    #   (OpenCV y-axis points down, so CW in image = positive angle here)
    #
    # Scissors:  tip points upper-left  → angle ≈ -145° (equiv. 215° or use ~-145)
    # Forceps:   tip points upper-left  → angle ≈ -155°
    # Tweezers:  tip points upper-left  → angle ≈ -160°
    tools = [
        # label,          cx_frac, cy_frac,  angle_deg,  note
        ("Scissors",       0.330,   0.880,   -146.21,   "Metzenbaum scissors"),
        ("Adson Forceps",  0.495,   0.840,   -144.63,   "Tissue forceps"),
        ("Fine Tweezers",  0.640,   0.860,   -146.29,   "Micro tweezers / needle holder"),
    ]

    overlay = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, cx_f, cy_f, angle, note in tools:
        cx = int(cx_f * w)
        cy = int(cy_f * h)

        # Draw center dot
        cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), 7, (40, 40, 40), 2)

        # Draw axes
        draw_axes(overlay, (cx, cy), angle, axis_len=90)

        # Label above center
        cv2.putText(overlay, label, (cx - 60, cy - 95), font, 0.65,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (cx - 60, cy - 95), font, 0.65,
                    (30, 30, 30), 1, cv2.LINE_AA)

    # Legend
    legend_y = 40
    for text, color in [("X — along tool (tip dir.)", (0,0,220)),
                        ("Y — in-plane perp.",         (0,200,0)),
                        ("Z — out of plane",            (220,80,0))]:
        cv2.putText(overlay, text, (20, legend_y), font, 0.6, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (20, legend_y), font, 0.6, color, 1, cv2.LINE_AA)
        legend_y += 28

    out_path = os.path.splitext(image_path)[0] + "_pose_axes.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")

    # Also display
    cv2.imshow("Surgical Tool Pose Axes", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default: look for the image in the working directory
        candidates = ["tools.jpg", "tools.png", "image.jpg", "image.png",
                      "surgical_tools.jpg", "surgical_tools.png"]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            print("Usage: python draw_pose_axes.py <image_path>")
            sys.exit(1)
        main(found)
    else:
        main(sys.argv[1])
