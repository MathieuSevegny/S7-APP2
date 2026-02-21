from matplotlib import pyplot as plt
import numpy as np
import skimage
from skimage.filters import sobel

from helpers import analysis
from helpers.dataset import Dataset
from scipy.signal import find_peaks

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
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(grayscale_image, cmap='gray')
        #plt.title('Image Grayscale')
        #plt.axis('off')
        #plt.subplot(1, 2, 2)

        #plt.imshow(magnitude_spectrum, cmap='gray')
        #plt.title('Magnitude Spectrum (log scale)')
        #plt.axis('off')
        #plt.show()

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

def calculate_ratio_high_low_frequency(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate ratio of high and low frequency from the vertical.
    
    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 1D array of shape (num_images,) containing the ratio of high-frequency to low-frequency components for each image.
    """
    ratios = np.zeros(len(rgb_images_data))
    for i, (image, _) in enumerate(rgb_images_data):
        grayscale_image = np.mean(image, axis=2)  # Convert to grayscale by averaging the RGB channels
        fft_image = np.fft.fft2(grayscale_image)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.abs(fft_shifted)
        low_freq = np.mean(magnitude_spectrum[:grayscale_image.shape[0]//2, :])
        high_freq = np.mean(magnitude_spectrum[grayscale_image.shape[0]//2:, :])
        ratios[i] = high_freq / (low_freq + 1e-8)  # Avoid division by zero
    return ratios

def vertical_horizontal_ratio(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate ratio of vertical and horizontal edges.
    
    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 1D array of shape (num_images,) containing the ratio of vertical to horizontal edges for each image.
    """
    ratios = np.zeros(len(rgb_images_data))
    for i, (image, _) in enumerate(rgb_images_data):
        grayscale_image = np.mean(image, axis=2)  # Convert to grayscale by averaging the RGB channels
        edges = sobel(grayscale_image)
        vertical_edges = np.sum(np.abs(edges[:, :-1] - edges[:, 1:]))  # Vertical edge strength
        horizontal_edges = np.sum(np.abs(edges[:-1, :] - edges[1:, :]))  # Horizontal edge strength
        ratios[i] = vertical_edges / (horizontal_edges + 1e-8)  # Avoid division by zero
    return ratios

def calculate_ratio_symmetry(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate the symmetry of each image in the dataset.
    
    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 1D array of shape (num_images,) containing the symmetry level of each image.
    """
    symmetry_levels = np.zeros(len(rgb_images_data))
    for i, (image, _) in enumerate(rgb_images_data):
        grayscale_image = np.mean(image, axis=2)  # Convert to grayscale by averaging the RGB channels
        flipped_image = np.fliplr(grayscale_image)  # Flip the image horizontally
        symmetry_levels[i] = np.sum(np.abs(grayscale_image - flipped_image)) / grayscale_image.size
    return symmetry_levels

def calculate_lab_b_peaks(rgb_images_data:Dataset) -> np.ndarray:
    """
    Calculate the number of peaks in the Lab color space's b channel for each image in the dataset.

    Args:
        rgb_images_data (np.ndarray): A 4D array of shape (num_images, height, width, 3) containing RGB pixel values.
    Returns:
        np.ndarray: A 1D array of shape (num_images,) containing the number of peaks in the Lab color space's b channel for each image.
    """
    spike_widths = np.zeros(len(rgb_images_data))
    for i, (image, _) in enumerate(rgb_images_data):
        # Convert RGB to Lab color space
        image_lab = skimage.color.rgb2lab(image / 255.0)
        scaled_lab = analysis.rescale_lab(image_lab, n_bins=256)
        peaks_b, _ = find_peaks(scaled_lab[:, :, 2].flatten())

        spike_widths[i] = len(peaks_b)

    return spike_widths