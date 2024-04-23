import os
from PIL import Image

def crop_images(input_folder, output_folder, size=(128, 128)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through all folders and subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Check if the file is an image
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                # Open the image
                img_path = os.path.join(root, file)
                img = Image.open(img_path)

                # Crop the image to specified size
                width, height = img.size
                left = (width - size[0]) // 2
                top = (height - size[1]) // 2
                right = (width + size[0]) // 2
                bottom = (height + size[1]) // 2
                img_cropped = img.crop((left, top, right, bottom))

                # Save the cropped image to the output folder
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                output_path = os.path.join(output_subfolder, file)
                img_cropped.save(output_path)

if __name__ == "__main__":
    # Set input and output folders
    input_folder = "/home/bwil0017/ir37_scratch/Stereo magnetograms vs hmi magnetograms"
    output_folder = "/home/bwil0017/ir37_scratch/Stereo magnetograms vs hmi magnetograms cropped"

    # Call the function to crop images
    crop_images(input_folder, output_folder)

