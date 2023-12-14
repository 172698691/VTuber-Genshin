import os
import cv2
import numpy as np
import time
import utils
import pickle
import mediapipe as mp
from argparse import ArgumentParser
from sock import Socket

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run():
    # Get operating system
    os = utils.get_os()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        if os == 'Windows':
            cap = cv2.VideoCapture(args.cam + cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(args.cam)


    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Setup face detection models
    
    # Import face detection model
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Import 3D face landmark detection model
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_model = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # TRUE: static image / False: real-time camera input
        refine_landmarks=True,  # Use the Attention Mesh model
        max_num_faces=1,  # Maximum number of faces to detect
        min_detection_confidence=0.5,  # Confidence threshold for detection, closer to 1 for higher accuracy
        min_tracking_confidence=0.5,  # Tracking threshold
    )
        
    # Establish a TCP connection to Unity
    if args.connect:
        address = ('127.0.0.1', 14514) # default port number 14514
        sock = Socket()
        sock.connect(address)

    ts = []
    frame_count = 0
    
    prev_pose = None
    pose_model = pickle.load(open('./model.pkl', 'rb'))

    while cap.isOpened():
        # Get frames
        ret, frame = cap.read()
        
        if not ret:
            print("End of video. Exiting...")
            break
    
        frame = cv2.flip(frame, 2)
        frame_RGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_count += 1

        # Send message data to Unity client
        if args.connect and frame_count > 60:
            sock.conv2msg()
            sock.send()

        t = time.time()

        # Loop
        # 1. Face detection, draw face and iris landmarks
        # 2. Pose estimation and stabilization (face + iris), calculate and calibrate data if error is low
        # 3. Data transmission with socket

        # Face detection on every frame
        face_detection_result = face_detector.process(frame_RGB)
        if face_detection_result.detections is None:
            continue
        else:
            face_detection = face_detection_result.detections[0]
            bboxC = face_detection.location_data.relative_bounding_box
            ih, iw, ic = frame_RGB.shape
            x0, y0, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            facebox = [x0, y0, x0 + w, y0 + h]

        # Face is detected
        if facebox is not None:
            # Mark face and iris on each frame
            
            # Process the image and obtain facial keypoints
            results = face_mesh_model.process(frame_RGB)
            if results.multi_face_landmarks is None:
                continue
            face_landmarks_3d = results.multi_face_landmarks[0]

            # Project to a 2D plane
            h, w, _ = frame.shape
            marks = []
            for landmark_3d in face_landmarks_3d.landmark:
                x = int(landmark_3d.x * w)
                y = int(landmark_3d.y * h)
                marks.append((x, y))
            marks = np.array(marks, dtype=np.int32)

            # Get indices of facial landmarks
            indices = utils.get_indices()
            x_l, y_l = marks[indices['left_iris']]
            x_r, y_r = marks[indices['right_iris']]
            
            pitch_pred, yaw_pred, roll_pred = utils.get_pose(pose_model, marks[indices['pose']])
            
            if args.connect:
                # head
                roll_r, pitch_r, yaw_r = np.degrees(roll_pred), np.degrees(pitch_pred), np.degrees(-yaw_pred)
                # Stabilization
                if prev_pose is not None:
                    roll = (roll_r + prev_pose[0]) / 2
                    pitch = (pitch_r + prev_pose[1]) / 2
                    yaw = (yaw_r + prev_pose[2]) / 2
                else:
                    roll, pitch, yaw = roll_r, pitch_r, yaw_r
                prev_pose = (roll_r, pitch_r, yaw_r)

                # eyes
                earLeft = utils.eye_aspect_ratio(marks[indices['left_eye']])
                earRight = utils.eye_aspect_ratio(marks[indices['right_eye']])
                eyeballX_l, eyeballY_l = utils.calculate_iris_position_ratio(marks, indices['left_eye'], indices['left_iris'])
                eyeballX_r, eyeballY_r = utils.calculate_iris_position_ratio(marks, indices['right_eye'], indices['right_iris'])
                eyeballX = (eyeballX_l + eyeballX_r) / 2
                eyeballY = (eyeballY_l + eyeballY_r) / 2

                # eyebrows
                barLeft = utils.brow_aspect_ratio(marks[indices['left_eyebrow']])
                barRight = utils.brow_aspect_ratio(marks[indices['right_eyebrow']])

                # mouth
                mouthWidthRatio = utils.mouth_distance(marks[indices['mouth_inner']]) / (facebox[2] - facebox[0])
                mouthOpen = utils.mouth_aspect_ratio(marks[indices['mouth_inner']])

                # Calibration before data transmission
                # eye openness
                eyeOpenLeft = utils.calibrate_eyeOpen(
                    earLeft, sock.eyeOpenLeftLast)
                eyeOpenRight = utils.calibrate_eyeOpen(
                    earRight, sock.eyeOpenRightLast)

                # eyeballs
                eyeballX, eyeballY = utils.calibrate_eyeball(
                    eyeballX, eyeballY)

                # eyebrows
                eyebrowLeft = utils.calibrate_eyebrow(
                    barLeft, sock.eyebrowLeftLast)
                eyebrowRight = utils.calibrate_eyebrow(
                    barRight, sock.eyebrowRightLast)

                # mouth width
                mouthWidth = utils.calibrate_mouthWidth(
                    mouthWidthRatio)

                # Update
                sock.update_all(roll, pitch, yaw, eyeOpenLeft, eyeOpenRight, eyeballX,
                                eyeballY, eyebrowLeft, eyebrowRight, mouthWidth, mouthOpen)

            # In debug mode, show the marks
            if args.debug:

                # Show facebox
                utils.draw_box(frame, [facebox])

                # Show iris
                if x_l > 0 and y_l > 0:
                    utils.draw_iris(frame, x_l, y_l, color=(0, 255, 255))
                if x_r > 0 and y_r > 0:
                    utils.draw_iris(frame, x_r, y_r, color=(0, 255, 255))

                # Show face landmarks
                frame = utils.draw_marks(frame, marks, indices)
                frame = utils.draw_axes(frame, pitch_pred, yaw_pred, roll_pred, marks[1][0], marks[1][1])

        dt = time.time() - t
        ts += [dt]
        FPS = int(1 / (np.mean(ts[-10:]) + 1e-6))
        print('\r', 'Time: %.3f' % dt, end=' ')

        utils.draw_FPS(frame, FPS)
        cv2.namedWindow("face", cv2.WINDOW_NORMAL)  # 使用cv2.WINDOW_NORMAL以支持手动调整大小
        cv2.resizeWindow("face", 512, 512)
        cv2.imshow("face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to exit
            break

    # Close all if program is terminated
    cap.release()
    if args.connect:
        sock.close()
    if args.debug:
        cv2.destroyAllWindows()
    print('Time: %.3f' % np.mean(ts))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cam", type=int,
                        help="specify the index of camera if there are multiple cameras",
                        default=0)
    parser.add_argument("--debug", action="store_true",
                        help="show image and marks",
                        default=False)
    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)
    parser.add_argument("--video", type=str,
                    help="specify the path to the video file (if not using camera)",
                    default=None)
    args = parser.parse_args()

    run()
