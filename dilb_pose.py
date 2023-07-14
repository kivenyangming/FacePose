import dlib
import cv2
from Config import get_head_pose

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""
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

"""


def draw_landmarks(src_img):
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks_part = []
    for face in faces:
        landmarks = predictor(gray, face)

        # 遍历关键点，并在图像上绘制出来
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_part.append([x, y])
            cv2.circle(src_img, (x, y), 2, (0, 255, 0), -1)
    return landmarks_part

def angle_xyz(point_x, point_y, point_z):
    """
      1. X 上负下正
      2. Y 左负右正
      3. Z 左正右负
      """
    if point_x > 5:
        point_x_status = "down"
    elif point_x < -5:
        point_x_status = "up"
    else:
        point_x_status = " "

    if point_y > 5:
        point_y_status = "right"
    elif point_y < -5:
        point_y_status = "left"
    else:
        point_y_status = " "

    if point_z > 5:
        point_z_status = "left"
    elif point_z < -5:
        point_z_status = "right"
    else:
        point_z_status = " "
    return point_x_status, point_y_status, point_z_status


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, src_img = cap.read()
        if not success:

            print("Ignoring empty camera frame.")
            continue
        else:
            landmarks_part = draw_landmarks(src_img)
            if len(landmarks_part) == 68:
                reprojectdst, euler_angle = get_head_pose(landmarks_part, 14)

                point_x = euler_angle[0][0]
                point_y = euler_angle[1][0]
                point_z = euler_angle[2][0]

                x_status, y_status, z_status = angle_xyz(point_x, point_y, point_z)

                cv2.putText(src_img, str("X: %.2f Status: %s" % (point_x, x_status)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(src_img, str('Y: %.2f Status: %s' % (point_y, y_status)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(src_img, str('Z: %.2f Status: %s' % (point_z, z_status)), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow("src_img", src_img)
            cv2.waitKey(1)

