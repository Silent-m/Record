import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

def restore_circle(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect the outer circle (label edge)
    outer_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                     param1=200, param2=100, minRadius=100, maxRadius=0)

    if outer_circles is not None:
        # Convert the (x, y) coordinates and radius of the outer circle to integers
        outer_circles = np.round(outer_circles[0, :]).astype("int")
        (x_outer, y_outer, r_outer) = outer_circles[np.argmax(outer_circles[:, 2])]

        # Dynamically set minRadius and maxRadius for the inner circle based on the outer circle size
        min_inner_radius = int(0.05 * r_outer)  # Inner hole is at least 5% of the outer radius
        max_inner_radius = int(0.2 * r_outer)   # Inner hole is at most 20% of the outer radius

        # Detect the inner circle (center hole)
        inner_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                         param1=200, param2=100, minRadius=min_inner_radius, maxRadius=max_inner_radius)

        if inner_circles is not None:
            # Convert the (x, y) coordinates and radius of the inner circle to integers
            inner_circles = np.round(inner_circles[0, :]).astype("int")
            (x_inner, y_inner, r_inner) = inner_circles[np.argmin(inner_circles[:, 2])]

            # Calculate the distortion (offset between the centers of the outer and inner circles)
            dx = x_outer - x_inner
            dy = y_outer - y_inner

            # Apply a perspective transformation to correct the distortion
            # Define the source points (original ellipse) and destination points (corrected circle)
            src_points = np.float32([[x_outer - r_outer, y_outer - r_outer],
                                     [x_outer + r_outer, y_outer - r_outer],
                                     [x_outer - r_outer, y_outer + r_outer],
                                     [x_outer + r_outer, y_outer + r_outer]])

            dst_points = np.float32([[x_inner - r_outer, y_inner - r_outer],
                                     [x_inner + r_outer, y_inner - r_outer],
                                     [x_inner - r_outer, y_inner + r_outer],
                                     [x_inner + r_outer, y_inner + r_outer]])

            # Compute the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply the transformation
            corrected_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

            # Crop the image to the corrected circle
            cropped_image = corrected_image[y_inner - r_outer:y_inner + r_outer,
                                           x_inner - r_outer:x_inner + r_outer]

            # Save the result
            cv2.imwrite(output_path, cropped_image)
            print(f"Result saved to {output_path}")
        else:
            print("Could not detect the inner circle.")
    else:
        print("Could not detect the outer circle.")

def main():
    # Hide the root Tkinter window
    Tk().withdraw()

    # Open a file selection dialog to choose the input image
    input_path = askopenfilename(title="Select an image file", filetypes=[("JPEG files", "*.jpg")])
    if not input_path:
        print("No file selected.")
        return

    # Open a file save dialog to choose the output image
    output_path = asksaveasfilename(title="Save the result as", defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
    if not output_path:
        print("No output file selected.")
        return

    # Restore the circle and save the result
    restore_circle(input_path, output_path)

if __name__ == "__main__":
    main()
