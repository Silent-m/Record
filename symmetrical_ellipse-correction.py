import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def open_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg")])
    return file_path

def is_circle(contour, tolerance=0.01):
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        circularity = minor_axis / major_axis
        return abs(1.0 - circularity) <= tolerance
    return False

def correct_symmetrical_ellipse(image, contour):
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    major_axis, minor_axis = max(axes), min(axes)
    
    estimated_diameter = major_axis  
    scale_factor = estimated_diameter / minor_axis
    
    height_padding = int((estimated_diameter - minor_axis) / 2)
    
    top_padding = height_padding
    bottom_padding = 0  
    left_right_padding = int((estimated_diameter - major_axis) / 2)
    
    edge_region = image[:5, :, :]
    avg_color = np.mean(edge_region, axis=(0, 1))
    
    new_height = image.shape[0] + top_padding + bottom_padding
    new_width = image.shape[1] + 2 * left_right_padding
    
    extended_image = np.full((new_height, new_width, 3), avg_color, dtype=np.uint8)
    
    extended_image[top_padding:top_padding + image.shape[0],
                   left_right_padding:left_right_padding + image.shape[1]] = image
    
    cv2.imwrite("debug_1_extended.jpg", extended_image)
    
    src_pts = np.array([[left_right_padding, top_padding],
                         [new_width - left_right_padding, top_padding],
                         [left_right_padding, new_height],
                         [new_width - left_right_padding, new_height]], dtype=np.float32)
    
    dst_pts = np.array([[0, top_padding],
                         [new_width, top_padding],
                         [0, new_height],
                         [new_width, new_height]], dtype=np.float32)
    
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_image = cv2.warpPerspective(extended_image, transform_matrix, (new_width, new_height))
    
    cv2.imwrite("debug_2_perspective_corrected.jpg", corrected_image)
    
    final_image = cv2.resize(corrected_image, (new_width, int(new_height * scale_factor)))
    
    cv2.imwrite("debug_3_final_corrected.jpg", final_image)
    
    return final_image

def process_image():
    file_path = open_image()
    if not file_path:
        print("No file selected.")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        return
    
    cv2.imwrite("debug_0_original.jpg", image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if not is_circle(largest_contour):
        print("Detected symmetrical ellipse. Applying correction...")
        corrected_image = correct_symmetrical_ellipse(image, largest_contour)
        output_path = os.path.join(os.path.dirname(file_path), "corrected.jpg")
        cv2.imwrite(output_path, corrected_image)
        print(f"Corrected image saved to {output_path}")
    else:
        print("Label is already circular. No correction needed.")

if __name__ == "__main__":
    process_image()
