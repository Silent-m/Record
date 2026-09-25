"""
Record Label Processor — Dark Background
For label-only images: circular label on black vinyl, rectangular frame.

Image structure (outer to inner):
    Optional white/grey border  (scanner background)
    -> Black vinyl rectangle
        -> Circular coloured label
            -> Small centre hole

Pipeline:
    load_image()
        -> crop_to_black()       # remove white border if present
        -> detect_label()        # bright circle on dark background
        -> detect_hole()         # small circle at label centre
        -> detect_rotation()     # angle from label region minus hole
        -> correct_rotation()    # rotate full image
        -> detect_label()        # re-detect centre after rotation (Stage 4b)
        -> crop_label()          # square crop around label
        -> resize_final()        # 1000x1000 output

Usage:
    python label_dark_bg.py
    (opens a file dialog to select an image)
"""

import os
from tkinter import Tk, filedialog

import cv2
import numpy as np



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINAL_SIZE        = (1000, 1000)
CROP_MARGIN       = 0.02          # extra margin around label when cropping
MAX_TILT          = 15.0          # maximum expected scan tilt in degrees
WHITE_BORDER_THR  = 200           # pixel value above which a row/col is "white"
WHITE_BORDER_FRAC = 0.80          # fraction of row/col that must be white to crop it
DEBUG             = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def open_image_dialog():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select label image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")]
    )
    return path or None


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"ERROR: Could not load image: {path}")
    return image


def save_image(image, original_path, suffix):
    directory = os.path.dirname(original_path)
    stem      = os.path.splitext(os.path.basename(original_path))[0]
    out_path  = os.path.join(directory, f"{stem}{suffix}.jpg")
    cv2.imwrite(out_path, image)
    print(f"  Saved: {out_path}")
    return out_path


def save_debug(image, original_path, tag):
    if DEBUG:
        save_image(image, original_path, f"_debug_{tag}")


# ---------------------------------------------------------------------------
# Stage 0 — Remove white/grey border around vinyl
# ---------------------------------------------------------------------------

def crop_to_black(image):
    """
    Remove white or grey scanner border around the black vinyl area.
    Scans inward from each edge; stops at the first row/col where
    fewer than WHITE_BORDER_FRAC of pixels exceed WHITE_BORDER_THR.
    Returns the cropped image (unchanged if no border found).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    def first_dark_row_from_top(g):
        for i in range(h):
            if np.mean(g[i, :] > WHITE_BORDER_THR) < WHITE_BORDER_FRAC:
                return i
        return 0

    def first_dark_row_from_bottom(g):
        for i in range(h - 1, -1, -1):
            if np.mean(g[i, :] > WHITE_BORDER_THR) < WHITE_BORDER_FRAC:
                return i
        return h - 1

    def first_dark_col_from_left(g):
        for j in range(w):
            if np.mean(g[:, j] > WHITE_BORDER_THR) < WHITE_BORDER_FRAC:
                return j
        return 0

    def first_dark_col_from_right(g):
        for j in range(w - 1, -1, -1):
            if np.mean(g[:, j] > WHITE_BORDER_THR) < WHITE_BORDER_FRAC:
                return j
        return w - 1

    top    = first_dark_row_from_top(gray)
    bottom = first_dark_row_from_bottom(gray)
    left   = first_dark_col_from_left(gray)
    right  = first_dark_col_from_right(gray)

    if top == 0 and bottom == h - 1 and left == 0 and right == w - 1:
        print("  No white border detected — image unchanged.")
        return image

    cropped = image[top:bottom + 1, left:right + 1]
    print(f"  White border removed: ({left},{top}) -> ({right},{bottom})  "
          f"new size {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


# ---------------------------------------------------------------------------
# Stage 1 — Detect label (bright circle on dark background)
# ---------------------------------------------------------------------------

def detect_label(image):
    """
    Detect the circular label on a dark (black vinyl) background.

    Strategy:
        1. Grayscale + blur.
        2. Otsu threshold: label is bright, vinyl is dark.
        3. Largest contour -> fitEllipse.

    Returns dict with keys:
        center       – (cx, cy)
        label_radius – in pixels
        ellipse      – raw fitEllipse result
    Returns None on failure.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Label is bright on dark background
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small gaps inside the label
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  ERROR: No contours found.")
        return None

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        print("  ERROR: Contour too small to fit ellipse.")
        return None

    ellipse          = cv2.fitEllipse(largest)
    center, axes, _  = ellipse
    label_radius     = int(max(axes) / 2)

    print(f"  Label: center=({center[0]:.0f}, {center[1]:.0f})  "
          f"radius≈{label_radius}px")

    return {
        "center"      : (int(center[0]), int(center[1])),
        "label_radius": label_radius,
        "ellipse"     : ellipse,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Detect hole radius (for masking during rotation detection)
# ---------------------------------------------------------------------------

def detect_hole(image, label):
    """
    Detect the small centre hole inside the label using HoughCircles.
    Falls back to 7.5% of label_radius if detection fails.
    """
    HOLE_RATIO = 0.075

    cx, cy       = label["center"]
    label_radius = label["label_radius"]

    # Crop to label area for faster, cleaner detection
    r   = label_radius
    x0  = max(0, cx - r);  y0 = max(0, cy - r)
    x1  = min(image.shape[1], cx + r)
    y1  = min(image.shape[0], cy + r)
    roi = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    min_r = max(5,  int(label_radius * 0.03))
    max_r = max(20, int(label_radius * 0.12))

    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=label_radius,
                               param1=80, param2=20,
                               minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        roi_cx = (x1 - x0) // 2
        roi_cy = (y1 - y0) // 2
        best = min(circles[0],
                   key=lambda c: (c[0] - roi_cx) ** 2 + (c[1] - roi_cy) ** 2)
        hole_radius = int(best[2])
        print(f"  Hole detected: radius={hole_radius}px")
        return hole_radius

    hole_radius = int(label_radius * HOLE_RATIO)
    print(f"  Hole not detected — using fallback radius={hole_radius}px")
    return hole_radius


# ---------------------------------------------------------------------------
# Stage 3 — Rotation detection (masked to label region minus hole)
# ---------------------------------------------------------------------------

def detect_rotation(image, label, hole_radius):
    """
    Detect rotation angle from text lines inside the label area.
    Same logic as record_label.py.
    """
    cx, cy       = label["center"]
    label_radius = label["label_radius"]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), label_radius, 255, -1)
    cv2.circle(mask, (cx, cy), hole_radius,    0, -1)

    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked   = cv2.bitwise_and(gray, gray, mask=mask)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)

    # Stretch contrast to full range to make faint text more visible
    min_val = np.min(enhanced[mask > 0])
    max_val = np.max(enhanced[mask > 0])
    if max_val > min_val:
        enhanced = np.clip((enhanced.astype(np.float32) - min_val) /
                           (max_val - min_val) * 255, 0, 255).astype(np.uint8)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    _, thresh = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=50, minLineLength=30, maxLineGap=10)
    if lines is None:
        print("  WARNING: No lines detected. Assuming 0°.")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -MAX_TILT <= a <= MAX_TILT and a != 0.0:
            angles.append(a)

    if not angles:
        print("  WARNING: No horizontal lines. Assuming 0°.")
        return 0.0

    # Find densest 1°-wide bin by scanning across the range
    best_center, best_count = 0.0, 0
    for c in np.arange(-MAX_TILT, MAX_TILT, 0.1):
        count = sum(1 for a in angles if abs(a - c) <= 0.5)
        if count > best_count:
            best_count, best_center = count, c
    tight = [a for a in angles if abs(a - best_center) <= 0.5]

    if len(tight) >= 4:
        angles = tight
    else:
        # Weak cluster — retry with sensitive parameters
        edges_s = cv2.Canny(thresh, 30, 100, apertureSize=3)
        lines_s = cv2.HoughLinesP(edges_s, 1, np.pi / 180,
                                  threshold=30, minLineLength=50, maxLineGap=5)
        if lines_s is not None:
            angles_s = []
            for l in lines_s:
                x1, y1, x2, y2 = l[0]
                a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -MAX_TILT <= a <= MAX_TILT and a != 0.0:
                    angles_s.append(a)
            if angles_s:
                median_s = float(np.median(angles_s))
                tight_s = [a for a in angles_s if abs(a - median_s) <= 2.0]
                if tight_s:
                    angles = tight_s
                    print("  Switched to sensitive parameters.")

    median_angle = -float(np.median(angles))
    print(f"  {len(angles)} lines  ->  median angle = {median_angle:.2f}°")
    return median_angle


# ---------------------------------------------------------------------------
# Stage 4 — Rotation correction
# ---------------------------------------------------------------------------

def correct_rotation(image, angle):
    """
    Rotate the FULL image so text becomes horizontal.
    detect_rotation() negates arctan2 output, so angle is already the correction to apply directly.
    """
    if abs(angle) < 0.1:
        print("  Angle negligible — skipping rotation.")
        return image

    h, w    = image.shape[:2]
    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    print(f"  Rotated by {angle:.2f}°")
    return rotated


# ---------------------------------------------------------------------------
# Stage 5 — Crop to label
# ---------------------------------------------------------------------------

def crop_label(image, label):
    cx, cy       = label["center"]
    label_radius = label["label_radius"]
    crop_radius  = int(label_radius * (1.0 + CROP_MARGIN))

    x0 = max(0, cx - crop_radius)
    y0 = max(0, cy - crop_radius)
    x1 = min(image.shape[1], cx + crop_radius)
    y1 = min(image.shape[0], cy + crop_radius)

    cropped = image[y0:y1, x0:x1]
    print(f"  Crop radius={crop_radius}px  "
          f"region=({x0},{y0})->({x1},{y1})  "
          f"size={cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


# ---------------------------------------------------------------------------
# Stage 6 — Final resize
# ---------------------------------------------------------------------------

def resize_final(image, size=FINAL_SIZE):
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    print(f"  Resized to {size[0]}x{size[1]}")
    return resized


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(path):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(path)}")
    print('='*60)

    image = load_image(path)
    if image is None:
        return
    save_debug(image, path, "0_original")

    # Stage 0 — Remove white border
    print("\n[Stage 0] Removing white border...")
    image = crop_to_black(image)
    save_debug(image, path, "1_cropped_border")

    # Stage 1 — Detect label
    print("\n[Stage 1] Detecting label...")
    label = detect_label(image)
    if label is None:
        print("FAILED: Could not detect label.")
        return

    # Stage 2 — Detect hole
    print("\n[Stage 2] Detecting centre hole...")
    hole_radius = detect_hole(image, label)

    # Debug: draw detections
    if DEBUG:
        dbg = image.copy()
        cx, cy = label["center"]
        cv2.ellipse(dbg, label["ellipse"], (255, 100, 0), 4)    # label  - blue
        cv2.circle(dbg, (cx, cy), hole_radius, (0, 0, 255), 4)  # hole   - red
        cv2.circle(dbg, (cx, cy), 8, (0, 255, 255), -1)         # center - yellow
        save_debug(dbg, path, "2_geometry")

    # Stage 3 — Detect rotation
    print("\n[Stage 3] Detecting rotation...")
    angle = detect_rotation(image, label, hole_radius)

    # Stage 4 — Rotate full image
    print("\n[Stage 4] Rotating...")
    rotated = correct_rotation(image, angle)
    save_debug(rotated, path, "3_rotated")

    # Stage 4b — Re-detect label centre after rotation
    print("\n[Stage 4b] Re-detecting label centre after rotation...")
    label_rotated = detect_label(rotated)
    if label_rotated is not None:
        delta_r = abs(label_rotated["label_radius"] - label["label_radius"])
        if delta_r < label["label_radius"] * 0.05:
            label = label_rotated
            print(f"  Centre updated to "
                  f"({label['center'][0]}, {label['center'][1]})")
        else:
            print(f"  WARNING: Unexpected radius change (delta={delta_r}px) "
                  f"— keeping original centre.")
    else:
        print("  WARNING: Re-detection failed — keeping original centre.")

    # Stage 5 — Crop
    print("\n[Stage 5] Cropping to label...")
    cropped = crop_label(rotated, label)
    save_debug(cropped, path, "4_cropped")

    # Stage 6 — Resize
    print("\n[Stage 6] Resizing...")
    final = resize_final(cropped)

    print("\n[Output]")
    save_image(final, path, "_result")

    print(f"\n{'='*60}")
    print(f"  Angle corrected: {angle:.2f}°")
    print('='*60)


def main():
    path = open_image_dialog()
    if path is None:
        print("No file selected.")
        return
    process_image(path)


if __name__ == "__main__":
    main()
