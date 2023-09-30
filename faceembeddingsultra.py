import cv2
from keras_facenet import FaceNet
import tensorflow as tf

# Suppress TensorFlow warning
tf.get_logger().setLevel('ERROR')

# Load the FaceNet model
embedder = FaceNet()

# Function to extract embeddings from an image
def extract_embeddings(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform face detection
    detections = embedder.extract(image, threshold=0.5)

    # Check if any faces were detected
    if detections:
        # Extract embeddings from the detections
        embeddings = [detection['embedding'] for detection in detections]
        return embeddings
    else:
        print(f"No faces detected in {image_path}.")
        return []

# Load the two input images
image1_path = "face1_0.jpg"
image2_path = "face2_0.jpg"

# Extract embeddings for both images
embeddings_image1 = extract_embeddings(image1_path)
embeddings_image2 = extract_embeddings(image2_path)

# Check if embeddings were successfully extracted
if embeddings_image1 and embeddings_image2:
    print("Embeddings for Image 1:")
    print(embeddings_image1)

    print("\nEmbeddings for Image 2:")
    print(embeddings_image2)
else:
    print("No embeddings extracted from one or both of the input images.")
