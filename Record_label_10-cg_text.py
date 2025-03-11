import cv2
import numpy as np
import pytesseract
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt
import os

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
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return mask
    return None

def detect_text_lines(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
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
    return np.median(angles) if angles else 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), 
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def crop_to_label(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]
    return None

def upscale_image(image, target_size=(1000, 1000)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    upscaled_image = pil_image.resize(target_size, Image.LANCZOS)
    return cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--psm 6 -l rus+eng'
    return pytesseract.image_to_string(binary, config=custom_config)

def save_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text saved to {output_path}")

def main():
    file_path = open_image()
    if not file_path:
        print("No file selected.")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        return
    
    mask = detect_label(image)
    if mask is None:
        print("Could not detect the label.")
        return
    
    lines = detect_text_lines(image, mask)
    if lines is None:
        print("Could not detect text lines.")
        return
    
    angle = calculate_rotation_angle(lines)
    print(f"Calculated rotation angle: {angle:.2f} degrees")
    rotated_image = rotate_image(image, -angle)
    
    cropped_image = crop_to_label(rotated_image, mask)
    if cropped_image is None:
        print("Could not crop the image.")
        return
    
    upscaled_image = upscale_image(cropped_image, target_size=(1000, 1000))
    output_img_path = os.path.join(os.path.dirname(file_path), "rezult.jpg")
    cv2.imwrite(output_img_path, upscaled_image)
    print(f"Image saved to {output_img_path}")
    
    extracted_text = extract_text(upscaled_image)
    output_text_path = os.path.join(os.path.dirname(file_path), "rezult.txt")
    save_text(extracted_text, output_text_path)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title("Rotated Image")
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title("Cropped Image")
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title("Final Processed Image")
    plt.imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
