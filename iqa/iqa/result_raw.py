import numpy as np
import cv2

def image_quality_metrics(I):
    # Calculate contrast
    def contrast(I):
        h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        mono = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        C = np.abs(cv2.filter2D(mono.astype(np.float32), -1, h, borderType=cv2.BORDER_REPLICATE))
        return C
    
    # Calculate saturation
    def saturation(I):
        R = I[:, :, 0]
        G = I[:, :, 1]
        B = I[:, :, 2]
        mu = (R + G + B) / 3
        C = np.sqrt(((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)
        return C
    
    # Calculate well-exposedness
    def well_exposedness(I):
        sig = 0.2
        R = np.exp(-0.5 * (I[:, :, 0] - 0.5) ** 2 / sig ** 2)
        G = np.exp(-0.5 * (I[:, :, 1] - 0.5) ** 2 / sig ** 2)
        B = np.exp(-0.5 * (I[:, :, 2] - 0.5) ** 2 / sig ** 2)
        C = R * G * B
        return C

    # Calculate metrics
    contrast_values = contrast(I)
    saturation_values = saturation(I)
    well_exposedness_values = well_exposedness(I)
    
    return contrast_values, saturation_values, well_exposedness_values

# Read an image from a file (adjust the file path as needed)
image_path = 'C:/Users/Duks/Desktop/iqa/house/A.jpg'
image = cv2.imread(image_path)

# Ensure the image is in RGB format (OpenCV loads images in BGR format)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Call the image_quality_metrics function to measure quality
contrast_values, saturation_values, well_exposedness_values = image_quality_metrics(image)

# Now you can analyze the quality metrics as needed