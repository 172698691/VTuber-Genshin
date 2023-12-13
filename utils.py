import numpy as np
from platform import system
import cv2
import pandas as pd


def get_os():
    """Get operating system"""
    return system()

def draw_box(image, boxes, box_color=(255, 255, 255)):
    """Draw square boxes on image, color=(b, g, r)"""
    for box in boxes:
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]), box_color, 3)
        
def draw_iris(frame, x, y, color=(255, 255, 255)):
    """Draw iris position, color=(b, g, r)"""
    cv2.line(frame, (x - 5, y), (x + 5, y), color)
    cv2.line(frame, (x, y - 5), (x, y + 5), color)

def draw_FPS(frame, FPS):
    """Draw FPS"""
    cv2.putText(frame, "FPS: %d" % FPS, (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

def eye_aspect_ratio(eye):
    """eye: np.array, shape (16, 2)"""
    assert eye.shape[0] % 2 == 0
    ear = 0
    for i in range (1, int(eye.shape[0]/2)):
        ear += np.linalg.norm(eye[i] - eye[eye.shape[0]-i])
    ear /= (2 * np.linalg.norm(eye[0] - eye[int(eye.shape[0]/2)]) + 1e-6)
    return ear

def calculate_iris_position_ratio(marks, eye_indices, iris_indices):
    """
    Calculate the relative position ratio (x_rate, y_rate) of the iris
    with respect to the eye region.

    Parameters:
    - marks: np.array, shape (N, 2), facial landmarks
    - eye_indices: list, indices of the eye region in marks
    - iris_indices: list, indices of the iris in marks

    Returns:
    - x_rate: how much iris is toward the left, 0 = totally left, 1 = totally right
    - y_rate: how much iris is toward the top, 0 = totally top, 1 = totally bottom
    """

    # Extract eye and iris coordinates
    eye_coords = marks[eye_indices]
    iris_coords = marks[iris_indices]

    # Calculate the bounding box of the eye region
    min_x = np.min(eye_coords[:, 0])
    max_x = np.max(eye_coords[:, 0])
    min_y = np.min(eye_coords[:, 1])
    max_y = np.max(eye_coords[:, 1])

    # Calculate the relative position of the iris
    x = iris_coords[0]
    y = iris_coords[1]
    x_rate = (x - min_x) / (max_x - min_x + 1e-6)
    y_rate = (y - min_y) / (max_y - min_y + 1e-6)

    return x_rate, y_rate

def brow_aspect_ratio(brow):
    """brow: np.array, shape (10, 2)"""
    mean_brow = np.array([(brow[i] + brow[brow.shape[0]-i-1]) / 2 for i in range(0, int(brow.shape[0]/2))])
    mid_point = (mean_brow[0] + mean_brow[4]) / 2
    bar = np.linalg.norm(mean_brow[2] - mid_point) / (np.linalg.norm(mean_brow[0] - mean_brow[4]) + 1e-6)
    return bar

def mouth_distance(mouth):
    """Calculate mouth distance"""
    assert mouth.shape[0] % 2 == 0
    return np.linalg.norm(mouth[0] - mouth[int(mouth.shape[0]/2)])

def mouth_aspect_ratio(mouth):
    """mouth: np.array, shape (20, 2)"""
    assert mouth.shape[0] % 2 == 0
    mar = 0
    for i in range(1, int(mouth.shape[0]/2)):
        mar += np.linalg.norm(mouth[i] - mouth[mouth.shape[0]-i])
    mar /= (2 * np.linalg.norm(mouth[0] - mouth[int(mouth.shape[0]/2)]) + 1e-6)
    return mar

def calibrate_eyeOpen(ear, eyeOpenLast):
    """Calibrate parameter eyeOpen"""
    flag = False  # jump out of current state
    
    if eyeOpenLast == 0.0:
        if ear > 0.25:
            flag = True
    if eyeOpenLast == 0.5:
        if abs(ear - 0.21) > 0.04:
            flag = True
    elif eyeOpenLast == 1.0:
        if abs(ear - 0.24) > 0.07:
            flag = True
    else:
        if ear < 0.17:
            flag = True

    if flag:
        if ear <= 0.22:
            eyeOpen = 0.0
        elif ear > 0.22 and ear <= 0.23:
            eyeOpen = 0.5
        elif ear > 0.23 and ear <= 0.27:
            eyeOpen = 1.0
        else:
            eyeOpen = 1.2
    else:
        eyeOpen = eyeOpenLast

    return eyeOpen


def calibrate_eyeball(eyeballX, eyeballY):
    """Calibrate parameters eyeballX, eyeballY"""
    return (eyeballX - 0.45) * 4.0, (eyeballY - 0.38) * 2.0


def calibrate_eyebrow(bar, eyebroLast):
    """Calibrate parameter eyebrow"""
    flag = False  # jump out of current state

    if eyebroLast == -1.0:
        if bar > 0.22:
            flag = True
    else:
        if bar < 0.2:
            flag = True

    if flag:
        if bar <= 0.225:
            eyebrow = -1.0
        else:
            eyebrow = 0.0
    else:
        eyebrow = eyebroLast

    return eyebrow


def calibrate_mouthWidth(mouthWidthRatio):
    """Calibrate parameter mouthWidth"""
    if mouthWidthRatio <= 0.32:
        mouthWidth = -0.5
    elif mouthWidthRatio > 0.32 and mouthWidthRatio <= 0.37:
        mouthWidth = 30.0 * mouthWidthRatio - 10.1
    else:
        mouthWidth = 1.0

    return mouthWidth


def get_pose(model, marks):
    """Predict pose from facial landmarks"""
    marks = marks.flatten()
    cols = []
    for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
        for dim in ('x', 'y'):
            cols.append(pos+dim)
            
    face_features_df = pd.DataFrame([marks], columns=cols)
    face_features_normalized = normalize(face_features_df)
    pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized.to_numpy()).ravel()
    return pitch_pred, yaw_pred, roll_pred


def normalize(poses_df):
    """Normalize facial landmarks"""
    normalized_df = poses_df.copy()
    
    for dim in ['x', 'y']:
        # Centerning around the nose 
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
        
        
        # Scaling
        diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = normalized_df[feature] / diff
    
    return normalized_df


def draw_marks(img, marks, indices):
    """Draw marks on image"""
    # Draw circles of different colors on the image
    new_img = img.copy()
    for i, landmark_2d in enumerate(marks):
        color = None
        if i in indices['left_eyebrow']:
            color = (255, 0, 0)  # Color for left eyebrow is red
        elif i in indices['right_eyebrow']:
            color = (0, 0, 255)  # Color for right eyebrow is blue
        elif i in indices['left_eye']:
            color = (0, 255, 0)  # Color for left eye is green
        elif i in indices['right_eye']:
            color = (0, 255, 255)  # Color for right eye is yellow
        elif i in indices['left_pupil']:
            color = (255, 0, 255)  # Color for left pupil is purple
        elif i in indices['right_pupil']:
            color = (255, 255, 0)  # Color for right pupil is cyan
        elif i in indices['mouth']:
            color = (225, 225, 225)  # Color for mouth is white
        if color is not None:
            cv2.circle(new_img, landmark_2d, 2, color, -1)
    return new_img


def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    """Draw axes on image"""
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img


def get_indices():
    """Get indices of facial landmarks"""
    # Define point indices for eyebrows, eyes, pupils, mouth
    left_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    right_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    left_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    right_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    left_iris_indices = 473
    right_iris_indices = 468
    left_pupil_indices = list(range(474, 478))
    right_pupil_indices = list(range(469, 473))
    mouth_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    mouth_inner_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    pose_indices = [1, 10, 33, 61, 199, 263, 291] # Nose, Forehead, Left eye, Left mouth, Chin, Right eye, Right mouth
    
    # Return a dictionary
    return {'left_eyebrow': left_eyebrow_indices, 'right_eyebrow': right_eyebrow_indices, 
            'left_eye': left_eye_indices, 'right_eye': right_eye_indices, 
            'left_iris': left_iris_indices, 'right_iris': right_iris_indices, 
            'left_pupil': left_pupil_indices, 'right_pupil': right_pupil_indices, 
            'mouth': mouth_indices, 'mouth_inner': mouth_inner_indices, 
            'pose': pose_indices}
