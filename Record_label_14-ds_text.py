import cv2
import numpy as np
from tkinter import Tk, filedialog
import pytesseract
import os

# Set the path to the Tesseract executable (update this if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def open_image():
    # Open a file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg")])
    return file_path

def detect_label(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to isolate the label
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the label
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return mask
    return None

def detect_text_lines(image, mask):
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                            minLineLength=50, maxLineGap=10)
    return lines

def calculate_rotation_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    # Calculate the median angle (to avoid outliers)
    median_angle = np.median(angles)
    return median_angle

def rotate_image(image, angle):
    # Get the image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def crop_to_label(image, mask):
    # Find the bounding box of the label
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    return None

def preprocess_image_for_ocr(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding for better binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_text_from_image(image):
    # Preprocess the image for OCR
    preprocessed_image = preprocess_image_for_ocr(image)
    
    # Perform OCR using Tesseract (Cyrillic and numbers only)
    custom_config = r'--oem 3 --psm 6 -l rus'  # Use only Cyrillic
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return text

def save_text_to_file(text, output_path):
    # Save the extracted text to a .txt file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Text saved to {output_path}")

def main():
    # Open the image
    file_path = open_image()
    if not file_path:
        print("No file selected.")
        return
    
    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        return
    
    # Detect the label
    mask = detect_label(image)
    if mask is None:
        print("Could not detect the label.")
        return
    
    # Detect text lines on the label
    lines = detect_text_lines(image, mask)
    if lines is None:
        print("Could not detect text lines.")
        return
    
    # Calculate the rotation angle
    angle = calculate_rotation_angle(lines)
    print(f"Calculated rotation angle: {angle:.2f} degrees")
    
    # Correct the angle and rotate the image
    corrected_angle = -angle  # Adjust to make text horizontal
    rotated_image = rotate_image(image, -corrected_angle)  # Invert direction
    
    # Crop the image to the label
    cropped_image = crop_to_label(rotated_image, mask)
    if cropped_image is None:
        print("Could not crop the image.")
        return
    
    # Extract text from the cropped image
    text = extract_text_from_image(cropped_image)
    print("Extracted Text:\n", text)
    
    # Save the extracted text to a .txt file
    output_text_path = os.path.join(os.path.dirname(file_path), "label_text.txt")
    save_text_to_file(text, output_text_path)

if __name__ == "__main__":
    main()