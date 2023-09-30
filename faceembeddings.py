import cv2
from keras_facenet import FaceNet
import tensorflow as tf

# Suppress TensorFlow warning
tf.get_logger().setLevel('ERROR')

# Load the FaceNet model
embedder = FaceNet()

# Load the image
image = cv2.imread("Bhushan\Bhushan4.jpg")

# Perform face detection
detections = embedder.extract(image, threshold=0.5)

# Check if any faces were detected
if detections:
    # Extract embeddings from the detections
    embeddings = [detection['embedding'] for detection in detections]
    print(embeddings)

    # 'embeddings' is now a list of face embeddings
else:
    print("No faces detected in the image.")
