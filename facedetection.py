import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_save_faces(image1, image2, output_filename1, output_filename2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces1):
        face = image1[y:y + h, x:x + w]
        cv2.imwrite(output_filename1.format(i), face)

    for i, (x, y, w, h) in enumerate(faces2):
        face = image2[y:y + h, x:x + w]
        cv2.imwrite(output_filename2.format(i), face)

    return output_filename1, output_filename2
