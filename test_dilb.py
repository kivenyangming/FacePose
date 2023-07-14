import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
image = cv2.imread("person.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)
landmarks_part = []
for face in faces:
    landmarks = predictor(gray, face)

    # 遍历关键点，并在图像上绘制出来
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_part.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)