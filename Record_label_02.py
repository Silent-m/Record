import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt

def open_image():
    # Create a Tkinter root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Bring the root window to the front
    root.lift()
    root.attributes('-topmost', True)  # Ensure the window stays on top
    root.after_idle(root.attributes, '-topmost', False)  # Reset after the dialog is shown

    # Open a file dialog to select a JPG file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg")])
    
    if file_path:  # Check if a file was selected
        # Open and display the image
        img = Image.open(file_path)
        plt.imshow(img)
        plt.axis('off')  # Hide the axis
        plt.show()

# Call the function to open the image
open_image()