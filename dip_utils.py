import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Tuple

def print_image(image: np.ndarray) -> None:
  """
  Displays a grayscale image using Matplotlib.

  Args:
      image (np.ndarray): The grayscale image data as a NumPy array.

  Returns:
      None
  """
  plt.figure()
  plt.imshow(image.astype(np.uint8), cmap='gray')
  plt.axis('off')
  plt.show()


def load_image(filename: str) -> np.ndarray:
  """
  Loads an image from a file and converts it to grayscale using OpenCV.

  Args:
      filename (str): The path to the input image file.

  Returns:
      np.ndarray: The grayscale image data as a NumPy array.
  """
  grayscale_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  if grayscale_image is None: 
    print(f"No image with filename '{filename}' found.")
  return grayscale_image
def save_image(image: np.ndarray, filename: str) -> None:
  """
  Saves a NumPy array representing an image to a file with the given filename.

  Args:
      image (numpy.ndarray): The NumPy array representing the image to be saved.
      filename (str): The name of the file to be saved.

  Returns:
      None
  """
  im = Image.fromarray(image)
  if im.mode != 'RGB':
      im = im.convert('RGB')
  im.save(filename)


def visualize_spectrum(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Computes the magnitude spectrum and phase spectrum of an input image.

  Args:
      image (np.ndarray): An M x N grayscale image.

  Returns:
      Tuple[np.ndarray, np.ndarray]: A tuple of two NumPy arrays representing the magnitude
      spectrum and phase spectrum of the input image, respectively. Both arrays have dimensions
      M x N and are in grayscale format.
  """
  # Convert the input image to float32 data type
  image = image.astype(np.float32)
  # Compute the 2D Fourier transform of the input image
  fourier_transform = np.fft.fft2(image)
  # Shift the zero-frequency component of the Fourier transform to the center of the spectrum
  shifted_transform = np.fft.fftshift(fourier_transform)
  # Compute the magnitude spectrum by taking the absolute value of the Fourier transform
  magnitude_spectrum = np.abs(shifted_transform)
  # Compute the phase spectrum by taking the complex angle of the Fourier transform
  phase_spectrum = np.angle(shifted_transform)
  return magnitude_spectrum, phase_spectrum


def plot_image_fft(image: np.ndarray, include_spatial: bool = False, label: str = "Image") -> None:
  """
  Plots the magnitude and phase of the Fourier transform of a given image.

  Args:
      image (np.ndarray): An M x N grayscale image.
      include_spatial (bool, optional): If True, includes the spatial domain in the side-by-side
      graphs. Defaults to False.
      label (str, optional): The label over the spatial domain image. Defaults to "Image"
  Returns:
      None
  """
  
  # Compute the magnitude and phase of the Fourier transform
  fft_magnitude, fft_phase = visualize_spectrum(image)

  # Plot the magnitude and phase spectra
  fig, axs = plt.subplots(1, 2+include_spatial, figsize=(8+4*include_spatial, 4))
  if include_spatial:
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(label)
    axs[0].axis("off")
  # Note: We are printing the log transform of the magnitude.
  axs[0+include_spatial].imshow(np.log(1 + fft_magnitude), cmap='gray')
  axs[0+include_spatial].set_title('Magnitude spectrum')
  axs[0+include_spatial].axis("off")
  axs[1+include_spatial].imshow(fft_phase, cmap='gray')
  axs[1+include_spatial].set_title('Phase spectrum')
  axs[1+include_spatial].axis("off")

  plt.show()


def save_image(image: np.ndarray, filename: str) -> None:
  """
  Saves a NumPy array representing an image to a file with the given filename.

  Args:
      image (numpy.ndarray): The NumPy array representing the image to be saved.
      filename (str): The name of the file to be saved.

  Returns:
      None
  """
  im = Image.fromarray(image)
  if im.mode != 'RGB':
      im = im.convert('RGB')
  im.save(filename)


def visualize_spectrum(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Computes the magnitude spectrum and phase spectrum of an input image.

  Args:
      image (np.ndarray): An M x N grayscale image.

  Returns:
      Tuple[np.ndarray, np.ndarray]: A tuple of two NumPy arrays representing the magnitude
      spectrum and phase spectrum of the input image, respectively. Both arrays have dimensions
      M x N and are in grayscale format.
  """
  # Convert the input image to float32 data type
  image = image.astype(np.float32)
  # Compute the 2D Fourier transform of the input image
  fourier_transform = np.fft.fft2(image)
  # Shift the zero-frequency component of the Fourier transform to the center of the spectrum
  shifted_transform = np.fft.fftshift(fourier_transform)
  # Compute the magnitude spectrum by taking the absolute value of the Fourier transform
  magnitude_spectrum = np.abs(shifted_transform)
  # Compute the phase spectrum by taking the complex angle of the Fourier transform
  phase_spectrum = np.angle(shifted_transform)
  return magnitude_spectrum, phase_spectrum
  

def plot_image_fft(image: np.ndarray, include_spatial: bool = False, label: str = "Image") -> None:
  """
  Plots the magnitude and phase of the Fourier transform of a given image.

  Args:
      image (np.ndarray): An M x N grayscale image.
      include_spatial (bool, optional): If True, includes the spatial domain in the side-by-side
      graphs. Defaults to False.
      label (str, optional): The label over the spatial domain image. Defaults to "Image"
  Returns:
      None
  """
  
  # Compute the magnitude and phase of the Fourier transform
  fft_magnitude, fft_phase = visualize_spectrum(image)

  # Plot the magnitude and phase spectra
  fig, axs = plt.subplots(1, 2+include_spatial, figsize=(8+4*include_spatial, 4))
  if include_spatial:
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(label)
    axs[0].axis("off")
  # Note: We are printing the log transform of the magnitude.
  axs[0+include_spatial].imshow(np.log(1 + fft_magnitude), cmap='gray')
  axs[0+include_spatial].set_title('Magnitude spectrum')
  axs[0+include_spatial].axis("off")
  axs[1+include_spatial].imshow(fft_phase, cmap='gray')
  axs[1+include_spatial].set_title('Phase spectrum')
  axs[1+include_spatial].axis("off")

  plt.show()

def motion_blur_transfer_fn(M: int, N: int, a: float, b: float) -> np.ndarray:
  """Compute the transfer function for motion blur given the image dimensions and blur parameters.

  Args:
      M (int): The height of the image.
      N (int): The width of the image.
      a (float): The horizontal blur parameter.
      b (float): The vertical blur parameter.

  Returns:
      np.ndarray: The transfer function for motion blur.
  """
  v, u = np.meshgrid(range(-N//2+1, N//2+1), range(-M//2+1, M//2+1))
  val = u*a+v*b
  sinc_val = np.sinc(np.pi*val)
  transfer_function = np.exp(-1j*np.pi*val)*sinc_val
  return transfer_function


def motion_blur(image: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
  """Apply motion blur to an image using the specified blur parameters.

  Args:
      image (np.ndarray): The input image as a 2D numpy array.
      a (float): The horizontal blur parameter.
      b (float): The vertical blur parameter.

  Returns:
      Tuple[np.ndarray, np.ndarray]: A tuple containing the filtered image and the transfer function.
  """
  M, N = image.shape
  filter_fn = motion_blur_transfer_fn(M, N, a, b)
  img_fft_shift = np.fft.fftshift(np.fft.fft2(image))
  filtered_image = np.fft.ifft2(np.fft.ifftshift(img_fft_shift * filter_fn)).real
  return filtered_image, filter_fn

  
def get_gaussian(M: int, N: int, sigma: float) -> np.ndarray:
  """Compute a 2D Gaussian filter with the given dimensions and standard deviation.

  Args:
      M (int): The height of the filter.
      N (int): The width of the filter.
      sigma (float): The standard deviation of the Gaussian.

  Returns:
      np.ndarray: A 2D numpy array representing the Gaussian filter.
  """
  v, u = np.meshgrid(range(-N//2+1, N//2+1), range(-M//2+1, M//2+1))
  gaussian = np.exp(-(u**2 + v**2) / (2*sigma**2))
  return gaussian


def apply_filter_fn(image: np.ndarray, filter_fn: np.ndarray) -> np.ndarray:
  """Apply a given filter to an image.

  Args:
      image (np.ndarray): The input image as a 2D numpy array.
      filter_fn (np.ndarray): The filter as a 2D numpy array.

  Returns:
      np.ndarray: The filtered image as a 2D numpy array.
  """
  assert image.shape == filter_fn.shape
  image_fft = np.fft.fftshift(np.fft.fft2(image))
  filtered_fft = image_fft * filter_fn
  filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
  return filtered_image.real