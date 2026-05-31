"""
Record Label Processor
Stage 1: Rotation correction for circular labels on white background.

Image structure (outer to inner):
    White background
    -> Black vinyl record  (~4200 px diameter)
        -> Coloured label  (~52% of record radius)
            -> Small hole  (~4% of record radius)

Pipeline:
    load_image()
        -> detect_record()       # finds record circle (black on white)
        -> estimate_label()      # label region = 52% of record radius
        -> detect_rotation()     # angle from label region of full image
        -> correct_rotation()    # rotate full image
        -> crop_label()          # crop to label (corners = real vinyl)
        -> resize_final()        # 1000x1000 output
        -> save_results()        # image + debug images

Usage:
    python record_label.py
    (opens a file dialog to select an image)
"""

import cv2
import numpy as np
import os
from tkinter import Tk, filedialog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINAL_SIZE        = (1000, 1000)  # output image size in pixels
CIRCULARITY_TOL   = 0.05          # tolerance for circle vs ellipse (5%)
DEBUG             = True          # save intermediate images

# Geometry ratios (relative to record radius)
LABEL_RATIO       = 0.524         # label radius / record radius
HOLE_RATIO        = 0.040         # hole radius  / record radius
CROP_MARGIN       = 0.02          # extra margin around label when cropping


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def open_image_dialog():
    """Open a file dialog and return the selected image path."""
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select record image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")]
    )
    return path or None


def load_image(path):
    """Load image from path. Returns numpy array (BGR) or None on failure."""
    image = cv2.imread(path)
    if image is None:
        print(f"ERROR: Could not load image: {path}")
    return image


def save_image(image, original_path, suffix):
    """Save image next to the original with a suffix appended to the name."""
    directory = os.path.dirname(original_path)
    stem      = os.path.splitext(os.path.basename(original_path))[0]
    out_path  = os.path.join(directory, f"{stem}{suffix}.jpg")
    cv2.imwrite(out_path, image)
    print(f"  Saved: {out_path}")
    return out_path


def save_debug(image, original_path, tag):
    """Save a debug image only when DEBUG is True."""
    if DEBUG:
        save_image(image, original_path, f"_debug_{tag}")


# ---------------------------------------------------------------------------
# Stage 1 – Record detection (black circle on white background)
# ---------------------------------------------------------------------------

def detect_record(image):
    """
    Detect the vinyl record (large black circle on white background).

    Strategy:
        1. Grayscale + blur.
        2. Simple threshold: white background -> record is dark.
        3. Largest contour = the record.
        4. Fit ellipse, classify as circle or ellipse.

    Returns dict with keys:
        center    – (cx, cy) in pixels
        axes      – (major, minor) in pixels
        angle     – ellipse tilt angle in degrees
        ellipse   – raw cv2.fitEllipse result
        is_circle – True if circularity within CIRCULARITY_TOL
    Returns None on failure.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Record is dark on white background
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
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

    ellipse         = cv2.fitEllipse(largest)
    center, axes, angle = ellipse
    major           = max(axes)
    minor           = min(axes)
    circularity     = minor / major
    is_circle       = circularity >= (1.0 - CIRCULARITY_TOL)

    print(f"  Record: center=({center[0]:.0f}, {center[1]:.0f})  "
          f"axes=({major:.0f}, {minor:.0f})  "
          f"circularity={circularity:.4f}  "
          f"-> {'CIRCLE' if is_circle else 'ELLIPSE'}")

    return {
        "center"   : (int(center[0]), int(center[1])),
        "axes"     : (major, minor),
        "angle"    : angle,
        "ellipse"  : ellipse,
        "is_circle": is_circle,
    }


# ---------------------------------------------------------------------------
# Stage 2 – Estimate label geometry from record
# ---------------------------------------------------------------------------

def estimate_label(record):
    """
    Estimate the label circle from the record geometry.

    For these records:
        label radius = LABEL_RATIO * record_radius  (= 0.524)
        hole  radius = HOLE_RATIO  * record_radius  (= 0.040)

    Returns dict with keys:
        center        – same as record center
        record_radius – in pixels
        label_radius  – in pixels
        hole_radius   – in pixels
    """
    major, _     = record["axes"]
    record_radius = int(major / 2)
    label_radius  = int(record_radius * LABEL_RATIO)
    hole_radius   = int(record_radius * HOLE_RATIO)
    cx, cy        = record["center"]

    print(f"  Record radius : {record_radius} px")
    print(f"  Label  radius : {label_radius} px  "
          f"(ratio {LABEL_RATIO})")
    print(f"  Hole   radius : {hole_radius} px  "
          f"(ratio {HOLE_RATIO})")

    return {
        "center"       : (cx, cy),
        "record_radius": record_radius,
        "label_radius" : label_radius,
        "hole_radius"  : hole_radius,
    }


# ---------------------------------------------------------------------------
# Stage 3 – Rotation detection (masked to label region)
# ---------------------------------------------------------------------------

def detect_rotation(image, geometry):
    """
    Detect the rotation angle of text lines inside the label area.

    The label region mask excludes:
        - the black vinyl grooves outside the label
        - the white background
        - the center hole

    Strategy:
        1. Circular mask = label area minus hole area.
        2. CLAHE -> Otsu threshold -> Canny -> Hough lines.
        3. Keep horizontal-ish lines (-45° to +45°).
        4. Return median angle.

    Returns angle in degrees (positive = CCW tilt).
    Returns 0.0 if detection fails.
    """
    cx, cy       = geometry["center"]
    label_radius = geometry["label_radius"]
    hole_radius  = geometry["hole_radius"]

    # Mask: label circle minus hole
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), label_radius, 255, -1)
    cv2.circle(mask, (cx, cy), hole_radius,    0, -1)

    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked   = cv2.bitwise_and(gray, gray, mask=mask)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)

    _, thresh = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=50, minLineLength=30, maxLineGap=10)

    if lines is None:
        print("  WARNING: No lines detected. Assuming 0° rotation.")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if not angles:
        print("  WARNING: No horizontal lines found. Assuming 0° rotation.")
        return 0.0

    median_angle = float(np.median(angles))
    print(f"  {len(angles)} lines used  ->  median angle = {median_angle:.2f}°")
    return median_angle


# ---------------------------------------------------------------------------
# Stage 4 – Rotation correction (full image)
# ---------------------------------------------------------------------------

def correct_rotation(image, angle):
    """
    Rotate the FULL image so text becomes horizontal.

    Rotating before cropping ensures label bounding box corners
    contain real vinyl surface, not artificially filled pixels.

    arctan2 returns positive for CCW lines -> apply -angle to correct.
    """
    if abs(angle) < 0.1:
        print("  Angle negligible — skipping rotation.")
        return image

    h, w    = image.shape[:2]
    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    print(f"  Rotated full image by {angle:.2f}°")
    return rotated


# ---------------------------------------------------------------------------
# Stage 5 – Crop to label
# ---------------------------------------------------------------------------

def crop_label(image, geometry):
    """
    Crop the rotated full image to a square centred on the label.

    Crop radius = label_radius + CROP_MARGIN.
    Because the full image was rotated first, corners contain real vinyl.
    """
    cx, cy       = geometry["center"]
    label_radius = geometry["label_radius"]
    crop_radius  = int(label_radius * (1.0 + CROP_MARGIN))

    x0 = max(0, cx - crop_radius)
    y0 = max(0, cy - crop_radius)
    x1 = min(image.shape[1], cx + crop_radius)
    y1 = min(image.shape[0], cy + crop_radius)

    cropped = image[y0:y1, x0:x1]
    print(f"  Crop radius : {crop_radius} px")
    print(f"  Crop region : ({x0},{y0}) -> ({x1},{y1})  "
          f"size = {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


# ---------------------------------------------------------------------------
# Stage 6 – Final resize
# ---------------------------------------------------------------------------

def resize_final(image, size=FINAL_SIZE):
    """Resize to standard output size using Lanczos interpolation."""
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    print(f"  Resized to {size[0]}x{size[1]}")
    return resized


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(path):
    """Run the full pipeline on a single image."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(path)}")
    print('='*60)

    # Load
    image = load_image(path)
    if image is None:
        return
    save_debug(image, path, "0_original")

    # Stage 1 – Detect record
    print("\n[Stage 1] Detecting record...")
    record = detect_record(image)
    if record is None:
        print("FAILED: Could not detect record.")
        return

    # Stage 2 – Estimate label geometry
    print("\n[Stage 2] Estimating label geometry...")
    geometry = estimate_label(record)

    # Debug: draw record ellipse, label circle, hole circle
    if DEBUG:
        dbg = image.copy()
        cx, cy = geometry["center"]
        cv2.ellipse(dbg, record["ellipse"], (0, 255, 0), 4)   # record - green
        cv2.circle(dbg, (cx, cy), geometry["label_radius"],
                   (255, 100, 0), 4)                           # label  - blue
        cv2.circle(dbg, (cx, cy), geometry["hole_radius"],
                   (0, 0, 255), 4)                             # hole   - red
        cv2.circle(dbg, (cx, cy), 8, (0, 255, 255), -1)       # center - yellow
        save_debug(dbg, path, "1_geometry")

    # Stage 3 – Detect rotation
    print("\n[Stage 3] Detecting rotation angle...")
    angle = detect_rotation(image, geometry)

    # Stage 4 – Rotate full image
    print("\n[Stage 4] Rotating full image...")
    rotated = correct_rotation(image, angle)
    save_debug(rotated, path, "2_rotated")

    # Stage 4b – Re-detect record center after rotation
    # The record may have shifted in the frame after rotation,
    # especially when it was not centered in a rectangular image.
    # We re-run detection on the rotated image to get the correct center.
    print("\n[Stage 4b] Re-detecting center after rotation...")
    record_rotated = detect_record(rotated)
    if record_rotated is not None:
        geometry_rotated = estimate_label(record_rotated)
        # Sanity check: label radius should not change significantly
        delta_r = abs(geometry_rotated["label_radius"] - geometry["label_radius"])
        if delta_r < geometry["label_radius"] * 0.05:
            geometry = geometry_rotated
            print(f"  Center updated to "
                  f"({geometry['center'][0]}, {geometry['center'][1]})")
        else:
            print(f"  WARNING: Re-detection gave unexpected radius "
                  f"(delta={delta_r}px) — keeping original center.")
    else:
        print("  WARNING: Re-detection failed — keeping original center.")

    # Stage 5 – Crop to label
    print("\n[Stage 5] Cropping to label...")
    cropped = crop_label(rotated, geometry)
    save_debug(cropped, path, "3_cropped")

    # Stage 6 – Resize
    print("\n[Stage 6] Resizing...")
    final = resize_final(cropped)

    # Save result
    print("\n[Output]")
    save_image(final, path, "_result")

    print(f"\n{'='*60}")
    print(f"  Measured angle     : {angle:.2f}°")
    print(f"  Applied correction : {angle:.2f}°")
    print('='*60)


def main():
    path = open_image_dialog()
    if path is None:
        print("No file selected.")
        return
    process_image(path)


if __name__ == "__main__":
    main()
