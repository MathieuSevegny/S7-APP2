import numpy as np

from helpers.dataset import Dataset

def calculate_noise(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate the noise level of each image in the dataset.
    
    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 1D array of shape (num_images,) containing the fft-based noise level of each image.
    """
    noise_levels = np.zeros(len(rgb_images_data))
    for i, (image, _) in enumerate(rgb_images_data):
        grayscale_image = np.mean(image, axis=2)  # Convert to grayscale by averaging the RGB channels
        fft_image = np.fft.fft2(grayscale_image)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.abs(fft_shifted)
        noise_levels[i] = np.sum(magnitude_spectrum) / (grayscale_image.size)
    return noise_levels

def calculate_contrast(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate the contrast level of each image in the dataset.
    
    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 1D array of shape (num_images,) containing the contrast level of each image.
    """
    contrast_levels = np.zeros(len(rgb_images_data))
    for i, (image, _) in enumerate(rgb_images_data):
        grayscale_image = np.mean(image, axis=2)  # Convert to grayscale by averaging the RGB channels
        contrast_levels[i] = np.max(grayscale_image) - np.min(grayscale_image)
    return contrast_levels

def calculate_most_common_color_in_top_left_corner(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate the most common color in the top-left corner of each image in the dataset.
    
    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 2D array of shape (num_images, 3) containing the most common RGB color in the top-left corner of each image.
    """
    most_common_colors = np.zeros((len(rgb_images_data), 3), dtype=int)
    for i, (image, _) in enumerate(rgb_images_data):
        top_left_corner = image[:10, :10]  # Consider a 10x10 region in the top-left corner
        reshaped_corner = top_left_corner.reshape(-1, 3)  # Reshape to a list of RGB values
        unique_colors, counts = np.unique(reshaped_corner, axis=0, return_counts=True)
        most_common_color = unique_colors[np.argmax(counts)]
        most_common_colors[i] = most_common_color
    return most_common_colors