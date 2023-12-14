import cv2
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable  # Make sure to install this library: pip install prettytable

def add_gaussian_noise(image, mean=0, sigma=0.01):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def recursive_filter(image, a, d):
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[2]):  
        for x in range(1, image.shape[0]):  
            for y in range(1, image.shape[1]):  
                filtered_image[x, y, i] = (1 - a*d) * image[x, y, i] + a*d * filtered_image[x - 1, y, i]

    return np.clip(filtered_image, 0, 255).astype(np.uint8)

def calculate_psnr(original, noisy, filtered):
    mse_noisy = np.mean((original - noisy) ** 2)
    mse_filtered = np.mean((original - filtered) ** 2)
    psnr_noisy = 20 * np.log10(255 / np.sqrt(mse_noisy))
    psnr_filtered = 20 * np.log10(255 / np.sqrt(mse_filtered))
    return psnr_noisy, psnr_filtered
# Specify the path to your image
image_path = 'C://Users//shaki//Project//2.jpg'

# Load the image
original_image = cv2.imread(image_path)

# Set the values for 'a' and 'd'
a = 0.2
d = 0.3  # Adjust this value based on experimentation
sigma = 0.01  # Adjust this value based on experimentation

# Create a PrettyTable for PSNR values
psnr_table = PrettyTable()
psnr_table.field_names = ["Image", "PSNR (dB)"]

# Add Gaussian noise with specified mean and sigma
noisy_image = add_gaussian_noise(original_image, sigma=sigma)
# Apply recursive filter
filtered_image = recursive_filter(noisy_image, a, d)
# Calculate PSNR
psnr_noisy, psnr_filtered = calculate_psnr(original_image, noisy_image, filtered_image)
# Add to the table
psnr_table.add_row(["Noisy Image", f"{psnr_noisy:.2f}"])
psnr_table.add_row(["Filtered Image", f"{psnr_filtered:.2f}"])

# Display the images and PSNR table
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')

# Noisy Image
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'Noisy Image\nPSNR: {psnr_noisy:.2f} dB')

# Recursive Filtered Image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'Recursive Filtered Image\nPSNR: {psnr_filtered:.2f} dB')

plt.show()

# Display the PSNR table
print(psnr_table)
