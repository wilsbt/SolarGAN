# Define paths to the folders containing the images
target = "/home/bwil0017/ir37_scratch/Stereo magnetograms vs hmi magnetograms cropped/test/test_output"
comparisons = ["/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Cropped_500000_1/test_input1_to_test_output/ITER0500000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Cropped_500000_1/test_input1_to_test_output/ITER0400000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Cropped_500000_1/test_input1_to_test_output/ITER0100000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Loss_50000_1/test_input1_to_test_output/ITER0050000",
               "/home/bwil0017/ir37/bwil0017/SolarGAN/Results/Loss_50_1/test_input1_to_test_output/ITER0000050"]
# Path to the folder containing images
folder_path = "/home/bwil0017/ir37_scratch/Stereo magnetograms vs hmi magnetograms cropped/test/test_output"

import os
from imageio import imread
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to load images from a folder
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = imread(os.path.join(folder_path, filename))  # Load images as grayscale
        if img is not None:
            images.append(img)
    return images

# Function to flatten and stack images into a 2D array
def preprocess_images(images):
    flattened_images = [img.flatten() for img in images]
    return np.stack(flattened_images)



# Load images
images = load_images(folder_path)

# Preprocess images
X = preprocess_images(images)

# Apply PCA
n_components = 40  # Number of principal components
pca = PCA(n_components=n_components).fit(X)
print(X[0].shape)
print(pca.transform(X))
print(pca.inverse_transform(pca.transform(X))[0].shape)






# # Plot the results
# plt.figure(figsize=(8, 6))
# for i in range(len(X_pca)):
#     plt.scatter(X_pca[i, 0], X_pca[i, 1], label=f"Image {i+1}")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA of Images")
# # plt.legend()
# plt.show()


# Function to plot an image
def plot_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Function to reconstruct an image from flattened data
def reconstruct_image(data, pca_model):
    reconstructed_data = pca_model.inverse_transform(data)
    return reconstructed_data.reshape(pca_model.mean_.shape)


# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(X[0].reshape(128,128), cmap='gray')
plt.title("Original Image")
plt.axis('off')
#
# # Plot the reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(pca.inverse_transform(pca.transform(X))[0].reshape(128,128), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')
#
plt.show()

images_gen = load_images(comparisons[0])
y = preprocess_images(images_gen)
losses = np.mean(np.abs(pca.transform(X) - pca.transform(y)), axis=1)
l1_losses = [np.mean(np.abs(images[i] - images_gen[i])) for i in range(len(images))]
print(np.corrcoef(losses, l1_losses))

plt.plot(losses)
plt.plot(l1_losses)
plt.show()







# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(y[5].reshape(128,128), cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Plot the reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(pca.inverse_transform(pca.transform(y))[5].reshape(128,128), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')

plt.show()
print(y[5].shape)
print((pca.components_).shape)

#
for i in range(10):
     # Plot the original image
     plt.subplot(1, 2, 1)
     plt.imshow(images[i], cmap='gray')
     plt.title(f"Original Image {losses[i]}")
     plt.axis('off')

     # Plot the reconstructed image
     plt.subplot(1, 2, 2)
     plt.imshow(images_gen[i], cmap='gray')
     plt.title("Reconstructed Image")
     plt.axis('off')

     plt.show()
