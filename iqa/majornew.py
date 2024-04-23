import cv2
import numpy as np
import os
import csv

def rgb_to_hsi(image):
    # Normalize pixel values to the range [0, 1]
    image_normalized = image.astype(np.float32) / 255.0

    # Extract R, G, B components
    R, G, B = image_normalized[:, :, 0], image_normalized[:, :, 1], image_normalized[:, :, 2]

    # Compute Intensity (I)
    I = (R + G + B) / 3.0

    # Compute Saturation (S)
    minimum = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 0.001) * minimum)

    # Compute Hue (H)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-10), -1.0, 1.0))
    H = np.degrees(theta)
    H[B > G] = 360 - H[B > G]

    return H, S, I


def calculate_entropy(intensity_channel):
    # Calculate histogram of intensity values
    hist, _ = np.histogram(intensity_channel, bins=256, range=(0, 1))

    # Compute probability distribution
    prob_distribution = hist / np.sum(hist)

    # Remove zero probabilities to avoid NaN in the entropy calculation
    non_zero_probs = prob_distribution[prob_distribution > 0]

    # Calculate entropy
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return entropy

def calculate_local_entropy(intensity_channel, window_size=3):
    height, width = intensity_channel.shape
    local_entropy = np.zeros((height, width))

    half_window = window_size // 2

    for i in range(half_window, height - half_window):
        for j in range(half_window, width - half_window):
            local_region = intensity_channel[i - half_window:i + half_window + 1,
                                             j - half_window:j + half_window + 1]
            hist, _ = np.histogram(local_region, bins=256, range=(0, 1))
            prob_distribution = hist / np.sum(hist)
            non_zero_probs = prob_distribution[prob_distribution > 0]
            local_entropy[i, j] = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return local_entropy

def calculate_rms_contrast(intensity_channel):
    # Calculate the standard deviation of the intensity channel
    std_intensity = np.std(intensity_channel)

    return std_intensity

#extra constrast function
def rms_contrast(intensity_channel):
       
    # Calculate RMS contrast
    contrast = np.sqrt(np.mean(intensity_channel**2))
    
    return contrast

def calculate_local_contrast(intensity_channel, window_size=3):
    height, width = intensity_channel.shape
    local_contrast = np.zeros((height, width))

    half_window = window_size // 2

    for i in range(half_window, height - half_window):
        for j in range(half_window, width - half_window):
            local_region = intensity_channel[i - half_window:i + half_window + 1,
                                             j - half_window:j + half_window + 1]
            local_contrast[i, j] = np.std(local_region)

    return local_contrast


def process_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB to HSI
    H, S, I = rgb_to_hsi(image_rgb)

    # Calculate RMS contrast using the Intensity (I) component
    rms_contrast_value = calculate_rms_contrast(I)

    # Calculate the mean value of the S component
    mean_saturation = np.mean(S)

    # Calculate entropy based on the Intensity (I) component
    entropy_I = calculate_entropy(I)

    # Calculate local entropy
    local_entropy_I = calculate_local_entropy(I, window_size=5)  # Adjust window size as needed

    # Calculate local contrast
    local_contrast_I = calculate_local_contrast(I, window_size=5)  # Adjust window size as needed

    return [image_path, rms_contrast_value, mean_saturation, entropy_I,
            local_entropy_I, local_contrast_I]

if __name__ == "__main__":
    input_folder = "./pics"  # Replace with the path to your input images folder
    output_csv = "output_results.csv"

    # Initialize a CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        csv_writer.writerow(["Image Path", "RMS Contrast", "Mean Saturation", "Entropy",
                             "Local Entropy", "Local Contrast"])

        # Iterate through all image files in the folder
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter by image file extensions
                image_path = os.path.join(input_folder, filename)
                results = process_image(image_path)

                # Write the results to the CSV file
                csv_writer.writerow(results)
    
    print(f"Processed {len(os.listdir(input_folder))} images and saved results in {output_csv}")