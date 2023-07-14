import cv2
import mediapipe as mp
import numpy as np

'相机内参矩阵'
img_size = (640, 480)
focal_length = img_size[0]
camera_center = (img_size[1] / 2, img_size[0] / 2)
cam_matrix = np.array([[2049.608299, 1.241852862, 1032.391255],
                       [0, 2066.791362, 550.6131349],
                       [0, 0, 1]], dtype="double")

'畸变矩阵'
dist_coeffs = np.array([0.108221558, -0.232697802, 0.002050653, -0.004714754, 0])

'头部三维通用模型关键点坐标'
object_pts_6 = np.array([
        (0.0, 0.0, 0.0),  # 鼻子
        (-125.0, 170.0, -135.0),  # 左眼
        (125.0, 170.0, -135.0),  # 右眼
        (0.0, 0.0, -125.0),  # 嘴巴
        (-135.0, 180.0, -135.0),  # 左耳
        (135.0, 180.0, -135.0)   # 右耳
    ], dtype=float) / 4.5

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                        [10.0, -10.0, 10.0],
                        [-10.0, 10.0, 10.0],
                        [-10.0, -10.0, 10.0]])

def get_face_info(image):

    img_copy = image.copy()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    box_info, facial = None, None
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            facial = detection.location_data.relative_keypoints

    return facial

def CoordinateTransformation(w, h, img_fa):
    # 左眼
    key_0_x, key_0_y = int(img_fa[0].x * h), int(img_fa[0].y * w)
    # 右眼
    key_1_x, key_1_y = int(img_fa[1].x * h), int(img_fa[1].y * w)
    # 鼻子
    key_2_x, key_2_y = int(img_fa[2].x * h), int(img_fa[2].y * w)
    # 嘴巴
    key_3_x, key_3_y = int(img_fa[3].x * h), int(img_fa[3].y * w)
    # 左耳
    key_4_x, key_4_y = int(img_fa[4].x * h), int(img_fa[4].y * w)
    # 右耳
    key_5_x, key_5_y = int(img_fa[5].x * h), int(img_fa[5].y * w)

    keys = np.array([
        (key_2_x, key_2_y),  # 鼻子
        (key_0_x, key_0_y),  # 左眼
        (key_1_x, key_1_y),  # 右眼
        (key_3_x, key_3_y),  # 嘴巴
        (key_4_x, key_4_y),  # 左耳
        (key_5_x, key_5_y),  # 右耳
    ])

    return keys

def get_head_pose(shape):
    """
    即图像坐标系中点的坐标从face_landmark_localization的检测结果抽取姿态估计需要的点坐标
    """
    image_pts = np.float32([shape[0], shape[1],
                        shape[2], shape[3], shape[4], shape[5]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts_6, image_pts, cam_matrix, dist_coeffs)

    """
    函数projectPoints根据所给的3D坐标和已知的几何变换来求解投影后的2D坐标
    """
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(4, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def draw_axis(img, euler_angle, center, size=80, thickness=3,
              angle_const=np.pi / 180, copy=False):
    if copy:
        img = img.copy()

    euler_angle *= angle_const
    sin_pitch, sin_yaw, sin_roll = np.sin(euler_angle)
    cos_pitch, cos_yaw, cos_roll = np.cos(euler_angle)

    axis = np.array([
        [cos_yaw * cos_roll,
         cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw],
        [-cos_yaw * sin_roll,
         cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll],
        [sin_yaw,
         -cos_yaw * sin_pitch]
    ])

    axis *= size
    axis += center

    axis = axis.astype(np.int)
    tp_center = tuple(center.astype(np.int))

    cv2.line(img, tp_center, tuple(axis[0]), (0, 0, 255), thickness)
    cv2.line(img, tp_center, tuple(axis[1]), (0, 255, 0), thickness)
    cv2.line(img, tp_center, tuple(axis[2]), (255, 0, 0), thickness)



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
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0,
                                         min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, src_img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            else:
                w, h, _ = src_img.shape  # 获取原始图像尺寸
                face_info = get_face_info(src_img)  # 五官坐标集
                landmarks_part = CoordinateTransformation(w, h, face_info)  # 转换各种坐标集格式便于读取
                if len(landmarks_part) == 6:
                    reprojectdst, euler_angle = get_head_pose(landmarks_part)

                    point_x = euler_angle[0][0]
                    point_y = euler_angle[1][0]
                    point_z = euler_angle[2][0]

                    x_status, y_status, z_status = angle_xyz(point_x, point_y, point_z)

                    cv2.putText(src_img, str("X: %.2f Status: %s" % (point_x, x_status)), (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    cv2.putText(src_img, str('Y: %.2f Status: %s' % (point_y, y_status)), (5, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    cv2.putText(src_img, str('Z: %.2f Status: %s' % (point_z, z_status)), (5, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    draw_axis(src_img, euler_angle, np.mean(landmarks_part,axis=0))

                cv2.imshow("src_img", src_img)
                cv2.waitKey(1)
