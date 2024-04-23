import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

# Define paths to the folders containing the images
target = "/home/bwil0017/ir37_scratch/Stereo magnetograms vs hmi magnetograms cropped/test/test_output"
comparisons = ["/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Cropped_500000_1/test_input1_to_test_output/ITER0500000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Cropped_500000_1/test_input1_to_test_output/ITER0400000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Cropped_500000_1/test_input1_to_test_output/ITER0100000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Loss_50000_1/test_input1_to_test_output/ITER0050000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Loss_50_1/test_input1_to_test_output/ITER0000050"]

for comparison in comparisons:
    # Get lists of file names in each folder
    image_files_1 = os.listdir(target)
    image_files_2 = os.listdir(comparison)


    # Assuming you have the necessary preprocessing functions defined
    def preprocess_image(image_path):
        # Load and preprocess the image
        image = imread(image_path)
        # Perform any necessary preprocessing (resizing, normalization, etc.)
        return image


    # Initialize list to store L1 losses
    l1_losses = []

    # Iterate through each pair of images
    for file_1, file_2 in zip(image_files_1, image_files_2):
        # Construct full paths to the images
        image_path_1 = os.path.join(target, file_1)
        image_path_2 = os.path.join(comparison, file_2)

        # Load and preprocess the images
        image_1 = preprocess_image(image_path_1)
        image_2 = preprocess_image(image_path_2)

        # Calculate L1 loss
        l1_loss = np.mean(np.abs(image_1 - image_2))

        # Store the L1 loss
        l1_losses.append(l1_loss)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
        norm_name = "L1"
        norm = np.abs(image_1 - image_2)
        loss = np.mean(norm)
        axs[0].imshow(image_1, cmap='gray')
        axs[0].set_title('Image 1')
        axs[1].imshow(norm, cmap='gray')
        axs[1].set_title(f'Absolute Difference {norm_name} {loss}')
        axs[2].imshow(image_2, cmap='gray')
        axs[2].set_title('Image 2')
        for ax in axs:
            ax.axis('off')  # Hide axes
        plt.show()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
        norm_name = "L1/2"
        norm = np.abs(image_1 ** (1 / 2) - image_2 ** (1 / 2)) ** (2)
        loss = np.mean(norm)
        axs[0].imshow(image_1, cmap='gray')
        axs[0].set_title('Image 1')
        axs[1].imshow(norm, cmap='gray')
        axs[1].set_title(f'Absolute Difference {norm_name} {loss}')
        axs[2].imshow(image_2, cmap='gray')
        axs[2].set_title('Image 2')
        for ax in axs:
            ax.axis('off')  # Hide axes
        plt.show()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
        norm_name = "L1 threshold 50"
        norm = np.abs(image_1 - image_2)
        norm[norm < 50] = 0
        loss = np.mean(norm)
        axs[0].imshow(image_1, cmap='gray')
        axs[0].set_title('Image 1')
        axs[1].imshow(norm, cmap='gray')
        axs[1].set_title(f'Absolute Difference {norm_name} {loss}')
        axs[2].imshow(image_2, cmap='gray')
        axs[2].set_title('Image 2')
        for ax in axs:
            ax.axis('off')  # Hide axes
        plt.show()