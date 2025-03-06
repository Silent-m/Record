import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt

def open_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg")])
    return file_path

def detect_label(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        size = max(w, h)
        x_center, y_center = x + w // 2, y + h // 2
        x_min = max(x_center - size // 2, 0)
        y_min = max(y_center - size // 2, 0)
        x_max = min(x_center + size // 2, image.shape[1])
        y_max = min(y_center + size // 2, image.shape[0])
        return x_min, y_min, x_max, y_max
    return None

def detect_text_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                            minLineLength=50, maxLineGap=10)
    return lines

def calculate_rotation_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    return -median_angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def main():
    file_path = open_image()
    if not file_path:
        print("No file selected.")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        return
    
    label_coords = detect_label(image)
    if label_coords is None:
        print("Could not detect the label.")
        return
    x_min, y_min, x_max, y_max = label_coords
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    lines = detect_text_lines(cropped_image)
    if lines is None:
        print("Could not detect text lines.")
        return
    
    angle = calculate_rotation_angle(lines)
    print(f"Rotation angle: {angle:.2f} degrees")
    rotated_image = rotate_image(cropped_image, angle)
    
    #resized_image = cv2.resize(rotated_image, (1000, 1000), interpolation=cv2.LANCZOS4)
    resized_image = cv2.resize(rotated_image, (1000, 1000), interpolation=cv2.INTER_LANCZOS4)

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Processed Image")
    plt.show()

if __name__ == "__main__":
    main()