"""
Envelope Processor
Stage 1: Rotation correction for scanned envelopes.

Detection strategy:
    From the edge image we can see:
    - Right edge:  strong, full, reliable
    - Bottom edge: strong, full, reliable  
    - Left edge:   partial but gives correct x position
    - Top edge:    invisible (blends with background)

    So we:
    1. Detect right edge  -> right_x, rotation angle
    2. Detect bottom edge -> bottom_y
    3. Detect left edge   -> left_x  (even partial line gives x position)
    4. Calculate width    = right_x - left_x
    5. Calculate top_y    = bottom_y - width  (square assumption)
    6. Four corners known -> rotate -> crop

Pipeline:
    load_image()
        -> detect_strong_edges()  # right + bottom + left via HoughLines
        -> correct_rotation()     # rotate full image
        -> re-detect after rotation
        -> crop_envelope()        # crop to rectangle + margin
        -> resize_final()         # longest side = 2000px
        -> save_results()

Usage:
    python envelope_processor.py
"""

import cv2
import numpy as np
import os
from tkinter import Tk, filedialog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINAL_LONG_SIDE  = 2000
CROP_MARGIN_PX   = 5    # small margin after edge expansion (pixels)
EDGE_EXPANSION   = 15   # expand detected edges outward by this many pixels
DEBUG            = True
_current_path    = ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def open_image_dialog():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select envelope image",
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


def save_debug(image, tag):
    if DEBUG:
        save_image(image, _current_path, f"_debug_{tag}")


# ---------------------------------------------------------------------------
# Helper: line intersection
# ---------------------------------------------------------------------------

def intersect_lines(rho1, theta1, rho2, theta2):
    A   = np.array([[np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]])
    b   = np.array([rho1, rho2])
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    x, y = np.linalg.solve(A, b)
    return (int(round(x)), int(round(y)))


def line_to_points(rho, theta, length=10000):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    return (int(x0 + length*(-b)), int(y0 + length*(a))), \
           (int(x0 - length*(-b)), int(y0 - length*(a)))


def rho_to_x(rho, theta):
    """
    For a near-vertical line, compute the x intercept at y=0.
    x = rho / cos(theta)  when theta near 0 or pi.
    """
    if abs(np.cos(theta)) < 1e-6:
        return None
    return int(round(rho / np.cos(theta)))


def rho_to_y(rho, theta):
    """
    For a near-horizontal line, compute the y intercept at x=0.
    y = rho / sin(theta)  when theta near pi/2.
    """
    if abs(np.sin(theta)) < 1e-6:
        return None
    return int(round(rho / np.sin(theta)))


# ---------------------------------------------------------------------------
# Stage 1 – Prepare edges
# ---------------------------------------------------------------------------

def prepare_edges(image):
    """
    Create edge image optimised for detecting envelope borders.
    Strong blur suppresses internal photo content.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    edges   = cv2.Canny(blurred, 20, 60)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges   = cv2.dilate(edges, kernel, iterations=2)
    return edges


# ---------------------------------------------------------------------------
# Stage 2 – Detect strong edges
# ---------------------------------------------------------------------------

def detect_strong_edges(image):
    """
    Detect right, bottom, and left envelope edges using HoughLines.

    From the edge image analysis:
    - Right edge:  strong full vertical line   -> gives right_x + angle
    - Bottom edge: strong full horizontal line -> gives bottom_y + angle
    - Left edge:   partial vertical line       -> gives left_x

    Top edge is invisible, so:
        width = right_x - left_x
        top_y = bottom_y - width  (square envelope assumption)

    Returns dict with corners and angle, or None on failure.
    """
    h, w  = image.shape[:2]
    edges = prepare_edges(image)
    save_debug(edges, "1a_edges")

    # Adaptive threshold based on image size
    threshold = int(min(h, w) * 0.25)
    print(f"  Hough threshold: {threshold}")

    # Use fine angular resolution (0.1°) for accurate angle measurement
    lines = cv2.HoughLines(edges, 1, np.pi / 1800, threshold=threshold)
    if lines is None:
        print("  ERROR: No lines detected.")
        return None
    print(f"  Lines detected: {len(lines)}")

    # Separate into near-horizontal and near-vertical
    # Use 20° tolerance to handle rotated envelopes
    angle_tol   = np.radians(20)
    horizontals = []
    verticals   = []

    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.pi/2) < angle_tol:
            horizontals.append((rho, theta))
        elif theta < angle_tol or theta > (np.pi - angle_tol):
            verticals.append((rho, theta))

    print(f"  Horizontal lines: {len(horizontals)}  "
          f"Vertical lines: {len(verticals)}")

    if not horizontals or not verticals:
        print("  ERROR: Missing horizontal or vertical edges.")
        return None

    # Bottom edge = horizontal line with largest rho (furthest down)
    bottom = max(horizontals, key=lambda x: x[0])

    # Right edge = rightmost vertical line that is NOT the scanner boundary
    # Exclude lines within the last 5% of image width (scanner/table edge)
    right_candidates = []
    for v in verticals:
        rho_v, theta_v = v
        if abs(np.cos(theta_v)) > 1e-6:
            x_pos = abs(rho_v / np.cos(theta_v))
            if x_pos < w * 0.95:
                right_candidates.append(v)
    if not right_candidates:
        right_candidates = verticals  # fallback
    right = max(right_candidates, key=lambda x: abs(x[0]))

    # Left edge = vertical line with smallest absolute rho (furthest left)
    # but exclude lines that are too close to the right edge
    right_x_approx = abs(right[0])
    left_candidates = [v for v in verticals
                       if abs(v[0]) < right_x_approx * 0.95]
    if left_candidates:
        left = min(left_candidates, key=lambda x: abs(x[0]))
        left_x = abs(int(round(left[0] / np.cos(left[1])))
                     if abs(np.cos(left[1])) > 1e-6
                     else left[0])
        print(f"  Left edge detected: rho={left[0]:.1f}  "
              f"theta={np.degrees(left[1]):.2f}°  x≈{left_x}")
    else:
        left   = None
        left_x = None
        print("  Left edge not detected — will estimate from bottom edge.")

    # Compute key positions
    bottom_y = rho_to_y(bottom[0], bottom[1])
    right_x  = abs(int(round(right[0] / np.cos(right[1])))
                   if abs(np.cos(right[1])) > 1e-6
                   else right[0])

    print(f"  Bottom edge: rho={bottom[0]:.1f}  "
          f"theta={np.degrees(bottom[1]):.3f}°  y≈{bottom_y}")
    print(f"  Right edge:  rho={right[0]:.1f}  "
          f"theta={np.degrees(right[1]):.3f}°  x≈{right_x}")

    if left_x is None:
        # Fallback: estimate left_x from the image — assume envelope
        # starts roughly where the background ends on the left side
        # Use a column projection to find leftmost non-background column
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray, (31, 31), 0)
        col_mean = np.mean(blurred, axis=0)
        bg_level = col_mean[0]   # leftmost column = background
        for i, val in enumerate(col_mean):
            if abs(val - bg_level) > 10:
                left_x = i
                break
        if left_x is None:
            left_x = 0
        print(f"  Left edge estimated from projection: x≈{left_x}")

    # Compute width and derive top_y (square assumption)
    width  = right_x - left_x
    top_y  = bottom_y - width

    print(f"  Computed: width={width}  top_y={top_y}  bottom_y={bottom_y}")

    # Sanity check: top_y should be within the image
    if top_y < 0:
        print(f"  WARNING: top_y={top_y} is above image. "
              f"Clamping to 0.")
        top_y = 0

    height = bottom_y - top_y
    ratio  = min(width, height) / max(width, height)
    print(f"  Size: {width}x{height}  ratio={ratio:.3f}")
    if ratio < 0.85:
        print("  WARNING: Aspect ratio suggests non-square envelope.")
    else:
        print("  Aspect ratio OK.")

    # Expand edges outward to compensate for Canny detecting
    # the inner edge of the border shadow rather than the outer edge
    left_x   = left_x   - EDGE_EXPANSION
    right_x  = right_x  + EDGE_EXPANSION
    top_y    = top_y    - EDGE_EXPANSION - 10   # top needs extra expansion
    bottom_y = bottom_y + EDGE_EXPANSION - 10   # bottom needs less expansion

    # Four corners
    tl = (left_x,  top_y)
    tr = (right_x, top_y)
    br = (right_x, bottom_y)
    bl = (left_x,  bottom_y)

    # Rotation angle from bottom edge
    # theta=pi/2 means perfectly horizontal -> deviation = angle
    angle_deg = np.degrees(bottom[1]) - 90.0
    print(f"  Rotation angle: {angle_deg:.3f}°")

    return {
        "corners": [tl, tr, br, bl],
        "angle"  : angle_deg,
        "width"  : width,
        "height" : height,
        "bottom" : bottom,
        "right"  : right,
        "left"   : left,
        "all_h"  : horizontals,
        "all_v"  : verticals,
    }


# ---------------------------------------------------------------------------
# Stage 3 – Rotation correction
# ---------------------------------------------------------------------------

def correct_rotation(image, angle):
    """Rotate the FULL image to make the envelope upright."""
    if abs(angle) < 0.05:
        print("  Angle negligible — skipping rotation.")
        return image
    h, w    = image.shape[:2]
    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    print(f"  Rotated full image by {-angle:.3f}°")
    return rotated


# ---------------------------------------------------------------------------
# Stage 4 – Crop to envelope
# ---------------------------------------------------------------------------

def crop_envelope(image, detection, margin_px=CROP_MARGIN_PX):
    """Crop using bounding box of the four corners + margin."""
    corners = detection["corners"]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    x0 = max(0, min(xs) - margin_px)
    y0 = max(0, min(ys) - margin_px)
    x1 = min(image.shape[1], max(xs) + margin_px)
    y1 = min(image.shape[0], max(ys) + margin_px)

    cropped = image[y0:y1, x0:x1]
    print(f"  Crop: ({x0},{y0}) -> ({x1},{y1})  "
          f"output = {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


# ---------------------------------------------------------------------------
# Stage 5 – Final resize
# ---------------------------------------------------------------------------

def resize_final(image, long_side=FINAL_LONG_SIDE):
    """Resize so the longest side = long_side, preserving aspect ratio."""
    h, w = image.shape[:2]
    if w >= h:
        new_w, new_h = long_side, int(h * long_side / w)
    else:
        new_h, new_w = long_side, int(w * long_side / h)
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_LANCZOS4)
    print(f"  Resized to {new_w}x{new_h}")
    return resized


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(path):
    global _current_path
    _current_path = path

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(path)}")
    print('='*60)

    image = load_image(path)
    if image is None:
        return
    save_debug(image, "0_original")

    # Stage 1 – Detect edges
    print("\n[Stage 1] Detecting envelope edges...")
    detection = detect_strong_edges(image)
    if detection is None:
        print("FAILED: Could not detect envelope.")
        return

    # Debug: draw detected lines and rectangle
    if DEBUG:
        dbg = image.copy()
        for rho, theta in detection["all_h"]:
            p1, p2 = line_to_points(rho, theta)
            cv2.line(dbg, p1, p2, (255, 255, 0), 1)
        for rho, theta in detection["all_v"]:
            p1, p2 = line_to_points(rho, theta)
            cv2.line(dbg, p1, p2, (0, 255, 255), 1)
        for key, color in [("bottom", (0,255,0)), ("right", (0,255,0))]:
            rho, theta = detection[key]
            p1, p2 = line_to_points(rho, theta)
            cv2.line(dbg, p1, p2, color, 3)
        if detection["left"] is not None:
            rho, theta = detection["left"]
            p1, p2 = line_to_points(rho, theta)
            cv2.line(dbg, p1, p2, (0, 165, 255), 3)
        pts = np.array(detection["corners"], dtype=np.int32)
        cv2.polylines(dbg, [pts], isClosed=True,
                      color=(255, 100, 0), thickness=3)
        for corner in detection["corners"]:
            cv2.circle(dbg, corner, 15, (0, 0, 255), -1)
        save_debug(dbg, "1_detection")

    # Stage 2 – Rotate
    print("\n[Stage 2] Rotating full image...")
    angle   = detection["angle"]
    rotated = correct_rotation(image, angle)
    save_debug(rotated, "2_rotated")

    # Stage 3 – Re-detect on rotated image
    print("\n[Stage 3] Re-detecting after rotation...")
    detection_rotated = detect_strong_edges(rotated)
    if detection_rotated is not None:
        w0 = detection["width"]
        w1 = detection_rotated["width"]
        if abs(w1 - w0) < w0 * 0.05:
            detection = detection_rotated
            print("  Detection updated.")
        else:
            print(f"  WARNING: width changed {w0}->{w1} — keeping original.")
    else:
        print("  WARNING: Re-detection failed — keeping original.")

    # Stage 4 – Crop
    print("\n[Stage 4] Cropping to envelope...")
    cropped = crop_envelope(rotated, detection)
    save_debug(cropped, "3_cropped")

    # Stage 5 – Resize
    print("\n[Stage 5] Resizing...")
    final = resize_final(cropped)

    print("\n[Output]")
    save_image(final, path, "_result")

    print(f"\n{'='*60}")
    print(f"  Measured angle     : {angle:.3f}°")
    print(f"  Applied correction : {-angle:.3f}°")
    print('='*60)


def main():
    path = open_image_dialog()
    if path is None:
        print("No file selected.")
        return
    process_image(path)
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
