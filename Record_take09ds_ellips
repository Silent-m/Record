import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

def restore_circle(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the cleaned image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Initialize variables for outer and inner circles
    outer_circle = None
    inner_circle = None

    # Debug: Draw all contours for visualization
    debug_image = image.copy()
    for i, contour in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(debug_image, center, radius, (0, 255, 0), 2)
        cv2.putText(debug_image, f"Contour {i}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the debug image with all contours
    cv2.namedWindow("Debug: All Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("Debug: All Contours", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Loop through contours to find the outer and inner circles
    for contour in contours:
        # Approximate the contour to a circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Assume the largest contour is the outer circle
        if outer_circle is None:
            outer_circle = (center, radius)
        else:
            # Assume the next largest contour within the outer circle is the inner circle
            if cv2.pointPolygonTest(np.array([outer_circle[0]]), center, False) > 0:
                inner_circle = (center, radius)
                break

    # Debug: Draw detected circles on the original image
    debug_image = image.copy()
    if outer_circle is not None:
        cv2.circle(debug_image, outer_circle[0], outer_circle[1], (0, 255, 0), 2)
    if inner_circle is not None:
        cv2.circle(debug_image, inner_circle[0], inner_circle[1], (0, 0, 255), 2)

    # Show the debug image with detected circles
    cv2.namedWindow("Debug: Detected Circles", cv2.WINDOW_NORMAL)
    cv2.imshow("Debug: Detected Circles", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if outer_circle is not None and inner_circle is not None:
        (outer_center, outer_radius) = outer_circle
        (inner_center, inner_radius) = inner_circle

        # Calculate the distortion (offset between the centers of the outer and inner circles)
        dx = outer_center[0] - inner_center[0]
        dy = outer_center[1] - inner_center[1]

        # Apply a perspective transformation to correct the distortion
        # Define the source points (original ellipse) and destination points (corrected circle)
        src_points = np.float32([[outer_center[0] - outer_radius, outer_center[1] - outer_radius],
                                 [outer_center[0] + outer_radius, outer_center[1] - outer_radius],
                                 [outer_center[0] - outer_radius, outer_center[1] + outer_radius],
                                 [outer_center[0] + outer_radius, outer_center[1] + outer_radius]])

        dst_points = np.float32([[inner_center[0] - outer_radius, inner_center[1] - outer_radius],
                                 [inner_center[0] + outer_radius, inner_center[1] - outer_radius],
                                 [inner_center[0] - outer_radius, inner_center[1] + outer_radius],
                                 [inner_center[0] + outer_radius, inner_center[1] + outer_radius]])

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the transformation
        corrected_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        # Crop the image to the corrected circle
        cropped_image = corrected_image[inner_center[1] - outer_radius:inner_center[1] + outer_radius,
                                       inner_center[0] - outer_radius:inner_center[0] + outer_radius]

        # Save the result
        cv2.imwrite(output_path, cropped_image)
        print(f"Result saved to {output_path}")
    else:
        print("Could not detect both outer and inner circles.")

def main():
    # Hide the root Tkinter window
    Tk().withdraw()

    # Open a file selection dialog to choose the input image
    input_path = askopenfilename(title="Select an image file", filetypes=[("JPEG files", "*.jpg")])
    if not input_path:
        print("No file selected.")
        return

    # Set the default output path to "rez.jpg" in the same directory as the input image
    output_path = os.path.join(os.path.dirname(input_path), "rez.jpg")

    # Restore the circle and save the result
    restore_circle(input_path, output_path)

if __name__ == "__main__":
    main()