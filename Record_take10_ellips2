import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
import matplotlib.pyplot as plt

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
        return largest_contour
    return None

def is_label_circular(contour):
    # Fit a minimum enclosing circle to the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    
    # Calculate the contour area
    contour_area = cv2.contourArea(contour)
    
    # Calculate the area of the minimum enclosing circle
    circle_area = np.pi * (radius ** 2)
    
    # Calculate the circularity ratio
    circularity = contour_area / circle_area
    
    # If the circularity is close to 1, the label is circular
    return circularity > 0.9  # Adjust the threshold as needed

def restore_label_to_round(image, contour):
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Get the ellipse parameters
    (center, axes, angle) = ellipse
    major_axis = max(axes)
    minor_axis = min(axes)
    
    # Calculate the scaling factor to make the ellipse a circle
    scale_factor = major_axis / minor_axis
    
    # Create a transformation matrix to "unwarp" the ellipse into a circle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    
    # Apply the transformation
    restored_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Crop the restored label
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = restored_image[y:y+h, x:x+w]
    
    return cropped_image

def save_image(image, output_path):
    # Save the image to the specified path
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def display_images(original, restored):
    # Display the original and restored images using matplotlib
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Restored Label")
    plt.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

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
    contour = detect_label(image)
    if contour is None:
        print("Could not detect the label.")
        return
    
    # Check if the label is already circular
    if is_label_circular(contour):
        print("The label is already circular. No correction needed.")
        restored_image = image  # Use the original image
    else:
        print("The label is not circular. Applying corrections.")
        restored_image = restore_label_to_round(image, contour)
    
    # Save the restored image
    output_path = os.path.join(os.path.dirname(file_path), "restored_label.jpg")
    save_image(restored_image, output_path)
    
    # Display the original and restored images
    try:
        # Try using OpenCV's imshow
        cv2.imshow("Original Image", image)
        cv2.imshow("Restored Label", restored_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        # Fallback to matplotlib if OpenCV's GUI is not available
        print("OpenCV GUI not available. Using matplotlib to display images.")
        display_images(image, restored_image)

if __name__ == "__main__":
    main()