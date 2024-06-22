import os
import shutil
import numpy as np

def split_data(source_folder, train_folder, test_folder, train_size=0.8):
    """ Split the data into train and test sets and move them to respective folders. """
    files = [file for file in os.listdir(source_folder) if file.endswith('.xml')]
    np.random.shuffle(files)
    train_count = int(len(files) * train_size)

    # Create target directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Move files
    for i, file in enumerate(files):
        if i < train_count:
            shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))
            # Move the corresponding image
            shutil.move(os.path.join(source_folder, file.replace('.xml', '.jpg')), os.path.join(train_folder, file.replace('.xml', '.jpg')))
        else:
            shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))
            # Move the corresponding image
            shutil.move(os.path.join(source_folder, file.replace('.xml', '.jpg')), os.path.join(test_folder, file.replace('.xml', '.jpg')))

# Usage
source_folder = 'images'  # Path to the folder where the original dataset is stored
train_folder = 'images/train'  # Path to the folder where the train dataset will be stored
test_folder = 'images/test'  # Path to the folder where the test dataset will be stored
split_data(source_folder, train_folder, test_folder)
