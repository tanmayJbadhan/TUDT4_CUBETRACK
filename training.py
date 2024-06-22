import pandas as pd
import numpy as np
import os
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.object_detector import DataLoader

# Specify paths
data_dir = 'CubeFace_Artist'  # Adjust this path
csv_file = 'label.csv'  # Adjust this path

# Load and process the CSV
df = pd.read_csv(csv_file)
df['filename'] = df['image_name'].apply(lambda x: os.path.join(data_dir, x))
df['class'] = 'Artist_Info'

# Convert coordinates and normalize
df['xmin'] = df['bbox_x']
df['xmax'] = df['bbox_x'] + df['bbox_width']
df['ymin'] = df['bbox_y']
df['ymax'] = df['bbox_y'] + df['bbox_height']

# Select necessary columns and create a DataLoader object
df = df[['filename', 'class', 'xmin', 'xmax', 'ymin', 'ymax']]
data = DataLoader.from_pandas(df, 'filename', ['xmin', 'ymin', 'xmax', 'ymax'], 'class')

# Set up the model to be trained
spec = model_spec.get('efficientdet_lite0')

# Train the model
model = object_detector.create(data, model_spec=spec, batch_size=8, train_whole_model=True, epochs=50)

# Export to TensorFlow Lite
model.export(export_dir='.', tflite_filename='artist_info_detector.tflite')

# Optional: Export to a label file
model.export(export_dir='.', label_filename='label_map.txt')

print("Model and labels have been saved.")
