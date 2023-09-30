import cv2
from mtcnn import MTCNN

mtcnn_detector = MTCNN()

def detect_and_save_faces_mtcnn(image1, image2, output_filename1, output_filename2):

    faces1 = mtcnn_detector.detect_faces(image1)
    faces2 = mtcnn_detector.detect_faces(image2)

    for i, face_info in enumerate(faces1):
        x, y, w, h = face_info['box']
        face = image1[y:y + h, x:x + w]
        cv2.imwrite(output_filename1.format(i), face)

    for i, face_info in enumerate(faces2):
        x, y, w, h = face_info['box']
        face = image2[y:y + h, x:x + w]
        cv2.imwrite(output_filename2.format(i), face)

    return output_filename1, output_filename2


