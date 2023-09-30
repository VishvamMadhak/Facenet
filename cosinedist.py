import cv2
from keras_facenet import FaceNet
import numpy as np
from numpy.linalg import norm
from facedetection import detect_and_save_faces 
from mtcnn import MTCNN
from mtcnndetection import detect_and_save_faces_mtcnn

# Load the FaceNet model
embedder = FaceNet()

# Function to extract embeddings from an image
def extract_embeddings(image_path):
    print(image_path)
    image = cv2.imread(image_path)
    cv2.imshow('frame' , image)
    cv2.waitKey(0)

    if image is None:
        print(f"Failed to load the image from {image_path}.")
        return []

    # Perform face detection
    detections = embedder.extract(image, threshold=0.9)
    print("this are the detections obtained: " , detections)

    # Check if any faces were detected
    if detections:
        # Extract embeddings from the detections
        embeddings = [detection['embedding'] for detection in detections]
        return embeddings
    else:
        print(f"No faces detected in {image_path}.")
        return []

# Function to calculate cosine distance
def findCosineDistance(vector_1, vector_2):
    a = np.matmul(np.transpose(vector_1), vector_2)
    b = np.matmul(np.transpose(vector_1), vector_1)
    c = np.matmul(np.transpose(vector_2), vector_2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Load images
image1 = cv2.imread('images.jpeg')
image2 = cv2.imread('e9d5eb2d312154ef4a646d160e8e5e33.jpg')

if image1 is None or image2 is None:
    print("Failed to load one or both of the images.")
    exit(1)

# Define the output filename format strings
output_filename1 = 'face1.jpg'
output_filename2 = 'face2.jpg'

# Detect and save faces
# image1_path, image2_path = detect_and_save_faces(image1, image2, output_filename1, output_filename2)
image1_path, image2_path = detect_and_save_faces_mtcnn(image1, image2, output_filename1, output_filename2)

# Extract embeddings for both images
embeddings_image1 = extract_embeddings(image1_path)
embeddings_image2 = extract_embeddings(image2_path)

# Check if embeddings were successfully extracted
if embeddings_image1 and embeddings_image2:
    print("Embeddings for Image 1:")
    print(embeddings_image1)

    print("\nEmbeddings for Image 2:")
    print(embeddings_image2)

    # Calculate cosine similarity
    cosine_dist = findCosineDistance(embeddings_image1[0], embeddings_image2[0]) 
    print("The cosine distance between them is:", cosine_dist)
else:
    print("No embeddings extracted from one or both of the input images.")
