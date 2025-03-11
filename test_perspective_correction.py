import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def open_image():
    # Initialize Tkinter
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    
    # Destroy the Tkinter root window after selection
    root.destroy()
    
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

def correct_perspective(image, contour, output_folder):
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    print(f"Ellipse center: {center}, axes: {axes}, angle: {angle}")
    
    # Get the bounding box of the ellipse
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    # Create a mask for the ellipse
    mask = np.zeros_like(image[:, :, 0])
    cv2.ellipse(mask, ellipse, 255, -1)
    
    # Crop the image to the bounding box
    cropped = image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    
    # Apply the mask to the cropped image
    masked_image = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)
    
    # Resize to fit the bounding box
    square_image = cv2.resize(masked_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # Save the intermediate image
    intermediate_path = os.path.join(output_folder, "_correct_perspective.jpg")
    cv2.imwrite(intermediate_path, square_image)
    print(f"Intermediate image saved to {intermediate_path}")
    
    return square_image

def hand_fan_stretch(image, ellipse, output_folder):
    # Unpack ellipse parameters
    (center, axes, angle) = ellipse
    major_axis = max(axes)
    minor_axis = min(axes)
    
    # Get the dimensions of the input image
    h, w = image.shape[:2]
    
    # Calculate the maximum stretching factor (at the top of the image)
    max_stretch_factor = major_axis / minor_axis
    
    # Create a mapping for hand-fan stretching
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        # Calculate the stretching factor for this row (increases as you move up)
        stretch_factor = 1 + (max_stretch_factor - 1) * (i / h)
        
        for j in range(w):
            # Apply horizontal stretching
            map_x[i, j] = j * stretch_factor
            map_y[i, j] = i
    
    # Calculate the new canvas width
    new_width = int(w * max_stretch_factor)
    
    # Apply the remapping
    stretched_image = cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Save the intermediate image
    intermediate_path = os.path.join(output_folder, "_hand_fan_stretch.jpg")
    cv2.imwrite(intermediate_path, stretched_image)
    print(f"Intermediate image saved to {intermediate_path}")
    
    return stretched_image

def vertical_stretch(image, output_folder):
    # Get the dimensions of the input image
    h, w = image.shape[:2]
    
    # Calculate the vertical stretch factor to make the canvas square
    stretch_factor = w / h
    
    # Resize the image vertically
    new_height = int(h * stretch_factor)
    stretched_image = cv2.resize(image, (w, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Save the intermediate image
    intermediate_path = os.path.join(output_folder, "_vertical_stretch.jpg")
    cv2.imwrite(intermediate_path, stretched_image)
    print(f"Intermediate image saved to {intermediate_path}")
    
    return stretched_image

def main():
    # Open the image
    file_path = open_image()
    if not file_path:
        print("No file selected.")
        return
    
    print(f"Selected file: {file_path}")
    
    # Create an output folder for intermediate images
    output_folder = os.path.dirname(file_path)
    print(f"Saving intermediate images to: {output_folder}")
    
    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        return
    
    print("Image loaded successfully.")
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to isolate the label
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return
    
    print(f"Found {len(contours)} contours.")
    
    # Assume the largest contour is the label
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if the label is circular
    if not is_circle(largest_contour):
        print("Label is not circular. Applying perspective correction...")
        
        # Step 1: Generate the temporary output image
        temp_image = correct_perspective(image, largest_contour, output_folder)
        
        if temp_image is not None:
            print("Temporary image generated successfully.")
            
            # Step 2: Apply hand-fan stretching
            ellipse = cv2.fitEllipse(largest_contour)
            fan_stretched_image = hand_fan_stretch(temp_image, ellipse, output_folder)
            
            # Step 3: Apply vertical stretching
            final_image = vertical_stretch(fan_stretched_image, output_folder)
            
            # Save the final corrected image
            output_path = os.path.join(output_folder, "rezult.jpg")
            cv2.imwrite(output_path, final_image)
            print(f"Final corrected image saved to {output_path}")
        else:
            print("Perspective correction failed.")
    else:
        print("Label is already circular.")

if __name__ == "__main__":
    main()