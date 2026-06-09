import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

# Camera settings
CAMERA_INDEX = 0

# Multi-Otsu configuration
NUM_THRESHOLDS = 3

# Morphological closing
CLOSE_KERNEL_SIZE = 31

# Ignore tiny contours (noise)
MIN_CONTOUR_AREA = 400


def contour_color(idx):
    hue = (idx * 37) % 180
    hsv = cv2.UMat(1, 1, cv2.CV_8UC3)
    hsv.get()[0, 0] = (hue, 220, 255)
    bgr = cv2.cvtColor(hsv.get(), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def colorize_regions(regions):
    palette = np.array(
        [
            [30, 30, 30],      # class 0 (darkest)
            [255, 0, 0],       # class 1
            [0, 255, 255],     # class 2
            [0, 255, 0],       # class 3 (brightest)
        ],
        dtype=np.uint8,
    )
    idx = np.clip(regions, 0, len(palette) - 1)
    return palette[idx]


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera")
        return

    print("RGB camera opened")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholds = threshold_multiotsu(blur, classes=NUM_THRESHOLDS + 1)
        regions = np.digitize(blur, bins=thresholds)
        region_viz = colorize_regions(regions)
        mask = np.where(regions >= 1, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        object_count = len(large_contours)

        display = frame.copy()
        for i, contour in enumerate(large_contours):
            cv2.drawContours(display, [contour], -1, contour_color(i), 2)
        cv2.putText(
            display,
            f"Objects detected: {object_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("RGB Camera", display)
        cv2.imshow("Multi-Otsu Regions", region_viz)
        cv2.imshow("Multi-Otsu Mask", mask)
        cv2.imshow("Closed Edges", closed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
