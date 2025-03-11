import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def open_image():
    # Open a file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg")])
    return file_path

def is_circle(contour, tolerance=0.01):
    # Fit an ellipse to the contour
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        
        # Calculate circularity (1.0 = perfect circle)
        circularity = minor_axis / major_axis
        return abs(1.0 - circularity) <= tolerance
    return False

def correct_perspective(image, contour):
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    
    # Get the bounding box of the ellipse
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    
    # Create a mask for the ellipse
    mask = np.zeros_like(image[:, :, 0])
    cv2.ellipse(mask, ellipse, 255, -1)
    
    # Crop the image to the bounding box
    cropped = image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    
    # Apply the mask to the cropped image
    masked_image = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)
    
    # Resize to a square to correct perspective
    target_size = max(w, h)
    square_image = cv2.resize(masked_image, (target_size, target_size), 
                             interpolation=cv2.INTER_LANCZOS4)
    
    # Create a mask for the corrected image
    corrected_mask = cv2.resize(mask_cropped, (target_size, target_size), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Resize the mask to match the original image size
    original_height, original_width = image.shape[:2]
    final_mask = cv2.resize(corrected_mask, (original_width, original_height), 
                            interpolation=cv2.INTER_NEAREST)
    
    return square_image, final_mask

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
        
        # Check if the label is circular
        if not is_circle(largest_contour):
            print("Label is not circular. Applying perspective correction...")
            corrected_image, corrected_mask = correct_perspective(image, largest_contour)
            return corrected_mask  # Return the corrected mask
        
        # If circular, proceed as usual
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return mask
    return None
 
def detect_ellipse(image, tolerance=0.01):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Hough Circle Transform to detect circles/ellipses
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=0
    )
    
    if circles is not None:
        # Convert the (x, y, radius) to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Filter circles based on 1% tolerance for ellipse detection
        filtered_circles = []
        for (x, y, r) in circles:
            # Calculate the aspect ratio (width/height)
            aspect_ratio = 1.0  # Assume perfect circle initially
            # For ellipses, you can use contour detection to calculate the actual aspect ratio
            # For now, we'll assume circles and filter based on radius
            if abs(aspect_ratio - 1.0) <= tolerance:
                filtered_circles.append((x, y, r))
        
        if filtered_circles:
            return filtered_circles
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

def upscale_image(image, target_size=(1000, 1000)):
    # Use OpenCV for resizing
    upscaled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    return upscaled_image

def save_image(image, output_path):
    # Save the image to the specified path
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

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
    
    # Upscale the cropped image to 1000x1000 pixels
    upscaled_image = upscale_image(cropped_image, target_size=(1000, 1000))
    
    # Save the upscaled image as rezult.jpg
    output_path = os.path.join(os.path.dirname(file_path), "rezult.jpg")
    save_image(upscaled_image, output_path)

if __name__ == "__main__":
    main()