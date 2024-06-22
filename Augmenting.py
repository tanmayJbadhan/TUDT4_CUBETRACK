import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def augment_images(source_folder, aug_folder, num_augmented_images=100):
    # Create the directory for augmented images if it doesn't exist
    if not os.path.exists(aug_folder):
        os.makedirs(aug_folder)

    datagen = ImageDataGenerator(
        rotation_range=20,         # Random rotations
        width_shift_range=0.2,     # Horizontal shift
        height_shift_range=0.2,    # Vertical shift
        zoom_range=0.2,            # Zooming
        horizontal_flip=True,      # Horizontal flip
        fill_mode='nearest',       # Filling in missing pixels after a shift or rotation
        brightness_range=[0.5, 1.5]  # Brightness changes
    )

    def change_color(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue_shift = np.random.uniform(-10, 10)
        sat_shift = np.random.uniform(0.5, 1.5)
        val_shift = np.random.uniform(0.5, 1.5)

        hsv = hsv.astype(np.float64)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_shift, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val_shift, 0, 255)
        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Load images and apply transformations
    for filename in os.listdir(source_folder):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.reshape((1,) + img.shape)  # reshape the image for the ImageDataGenerator

        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_img = batch[0].astype(np.uint8)
            augmented_img = change_color(augmented_img)
            augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(aug_folder, f'aug_{filename}_{i}.jpg')
            cv2.imwrite(output_path, augmented_img)
            i += 1
            if i >= num_augmented_images:
                break  # break the loop to avoid generating infinite images

# Define paths
source_folder = 'CubeFace_Artist'
aug_folder = 'CubeFace_Augmented'

# Call the function
augment_images(source_folder, aug_folder, num_augmented_images=10)  # Generate 10 augmented images per original image

# Show some augmented images
plt.figure(figsize=(10, 10))
for i, file_name in enumerate(os.listdir(aug_folder)[:9]):
    img_path = os.path.join(aug_folder, file_name)
    img = plt.imread(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
plt.show()
