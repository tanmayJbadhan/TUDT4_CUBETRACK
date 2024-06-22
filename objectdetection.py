import cv2
import mediapipe as mp

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Objectron with adjusted confidence parameters
objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=5,
                                   min_detection_confidence=0.3,  # Lowered confidence
                                   min_tracking_confidence=0.3,   # Lowered confidence
                                   model_name='Cup')  # Using 'Cup' model

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect objects
    results = objectron.process(image_rgb)

    # Debugging output
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
                image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
    else:
        print("No objects detected")

    # Display the resulting frame
    cv2.imshow('MediaPipe Objectron - Bottle Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
