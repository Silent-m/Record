import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract
import matplotlib.pyplot as plt

def open_and_correct_image():
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    # Open a file dialog to select a JPG file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg")])
    
    if file_path:  # Check if a file was selected
        # Open image using OpenCV
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find text lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = (theta - np.pi / 2) * (180 / np.pi)  # Convert radians to degrees
                angles.append(angle)
            
            # Calculate average angle
            avg_angle = np.mean(angles)
        else:
            avg_angle = 0  # Default to no rotation if no lines are detected
        
        # Rotate image
        rotated = Image.open(file_path)
        rotated = ImageOps.exif_transpose(rotated)  # Handle EXIF orientation
        rotated = rotated.rotate(-avg_angle, expand=True)
        
        # Show the corrected image
        plt.imshow(rotated)
        plt.axis('off')
        plt.show()

# Run the function
open_and_correct_image()
