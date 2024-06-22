import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

def create_tf_example(group, path):
    # Correctly handle the filename and ensure it is just a string
    filename = group['filename'].values[0]  # Extract filename
    image_path = os.path.join(path, filename)
    print("Attempting to open image path:", image_path)  # Debug for path correctness

    # Open the image file
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_image_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_image_io)
    width, height = image.size

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    def class_text_to_int(row_label):
        label_map = {
        'Artist_Info': 1,
        'Info': 2,
        'Story': 3,
        'Like/Dislike': 4,
        'Star': 5
    }
        return label_map[row_label]

    for index, row in group.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_csv_to_tfrecord(csv_input, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)
    grouped = examples.groupby('filename')
    for filename, group in grouped:
        print("Processing file:", filename)  # Debugging print statement
        tf_example = create_tf_example(group, image_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


# Example usage:
csv_input = 'train_labels.csv'
image_dir = 'images/train'
output_path = 'train.record'
convert_csv_to_tfrecord(csv_input, image_dir, output_path)

csv_input = 'test_labels.csv'
image_dir = 'images/test'
output_path = 'test.record'
convert_csv_to_tfrecord(csv_input, image_dir, output_path)
