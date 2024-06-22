import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('cube_detection_model')

def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at path: {image_path}")
        return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (image_size, image_size))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image, image_expanded

def draw_bounding_box(image, predictions):
    h, w, _ = image.shape
    start_point = (int(predictions[0] * w), int(predictions[1] * h))
    end_point = (int(predictions[2] * w), int(predictions[3] * h))
    print(f"Bounding Box Coordinates: Start: {start_point}, End: {end_point}")  # Debugging output
    cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    cv2.putText(image, 'Cube Face', (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def test_model_on_image(model, image_path):
    image_size = 128
    original_image, input_image = preprocess_image(image_path, image_size)
    if original_image is None or input_image is None:
        return
    
    # Run inference
    preds = model.predict(input_image)
    preds = preds[0]  # Get the predictions for the first image in the batch
    print(f"Raw Predictions: {preds}")  # Debugging output

    # Draw bounding box
    image_with_box = draw_bounding_box(original_image, preds)
    
    # Display the result
    plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test the model on a given image
image_path = 'R.jpEg'
test_model_on_image(model, image_path)
