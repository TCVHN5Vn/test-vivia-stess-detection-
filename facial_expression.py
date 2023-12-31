import cv2,dlib
import mediapipe as mp
import numpy as np
from custom.core import extract_faces, tflite_inference
from deepface import DeepFace
import os
from keras.models import load_model
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array


mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

labels = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]

fast_model = True
slow_model_every_x = 5



def recognize_expression_image(image):
    image = cv2.imread(image)
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean = np.array([0.57535914, 0.44928582, 0.40079932])
    std = np.array([0.20735591, 0.18981615, 0.18132027])

    if fast_model:
        model_path = "model/efficient_face_model.tflite"
    else:
        model_path = "model/dlg_model.tflite"

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:

        results = face_detection.process(frame_rgb)

        face_frames = extract_faces(frame_rgb, results, x_scale=1.2, y_scale=1.2)

        if face_frames:
            face_frame = cv2.resize(face_frames[0], (224, 224))

            if fast_model:
                face_frame = face_frame / 255
                face_frame -= mean
                face_frame /= std
                face_frame = np.moveaxis(face_frame, -1, 0)

                outputs = tflite_inference(face_frame, model_path)
                outputs = outputs[0]
                expression_id = np.argmax(outputs)

                return labels[expression_id]
def extract_faces(frame, results, x_scale=1.2, y_scale=1.2):
    face_frames = []
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y, w, h = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), \
                     int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])
        x_offset = int(w * (x_scale - 1) / 2)
        y_offset = int(h * (y_scale - 1) / 2)
        x1, y1 = max(x - x_offset, 0), max(y - y_offset, 0)
        x2, y2 = min(x + w + x_offset, frame.shape[1]), min(y + h + y_offset, frame.shape[0])
        face_frame = frame[y1:y2, x1:x2]
        face_frames.append(face_frame)
    return face_frames

def get_dominant_emotions(emotion_frames,video_path):
    num_parts = int(get_video_time(video_path))
    frames_per_part = len(emotion_frames) // num_parts

    if len(emotion_frames) == 0:
        return []

    parts = [emotion_frames[i:i + frames_per_part] for i in range(0, len(emotion_frames), frames_per_part)]
    dominant_emotions = []

    for part in parts:
        dominant_emotion = max(set(part), key=part.count)
        dominant_emotions.append(dominant_emotion)

    return dominant_emotions



def recognize_expression_video(video_file):
    cap = cv2.VideoCapture(video_file)
    mean = np.array([0.57535914, 0.44928582, 0.40079932])
    std = np.array([0.20735591, 0.18981615, 0.18132027])

    if fast_model:
        model_path = "model/efficient_face_model.tflite"
    else:
        model_path = "model/dlg_model.tflite"

    emotion_frames = []  # List to store emotions for each frame

    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as face_detection:

            results = face_detection.process(frame_rgb)
            try:
                face_frames = extract_faces(frame_rgb, results, x_scale=1.2, y_scale=1.2)
            except FaceDetectionError:
    # Handle face detection errors
                emotion_frames.append("Error: Face not detected")
            except FaceExtractionError:
    # Handle face extraction errors
                emotion_frames.append("Error: Failed to extract face")

            if face_frames:
                face_frame = cv2.resize(face_frames[0], (224, 224))

                if fast_model:
                    face_frame = face_frame / 255
                    face_frame -= mean
                    face_frame /= std
                    face_frame = np.moveaxis(face_frame, -1, 0)

                    outputs = tflite_inference(face_frame, model_path)
                    outputs = outputs[0]
                    expression_id = np.argmax(outputs)

                    emotion_frames.append(labels[expression_id])  # Append emotion to the list

        frame_count += 1

        # If using the slow model, only run the model every slow_model_every_x frames
        if not fast_model and frame_count % slow_model_every_x == 0:
            model_path = "model/dlg_model.tflite"

    # Calculate the percentage of each emotion in the video
    total_frames = frame_count
    emotion_percentages = {emotion: '{:.2}'.format(emotion_frames.count(emotion) / total_frames) for emotion in labels}

    return [emotion_frames,emotion_percentages]


def count_head_movements(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    count_left = 0
    count_right = 0
    count_up = 0
    count_down = 0
    prev_direction = "forward"

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                direction = ""

                if y < -7:
                    direction = "left"
                elif y > 7:
                    direction = "right"
                elif x < -7:
                    direction = "down"
                elif x > 7:
                    direction = "up"
                else:
                    direction = "forward"

                if prev_direction != direction:
                    if direction == "left" and prev_direction == "forward":
                        count_left += 1
                    elif direction == "right" and prev_direction == "forward":
                        count_right += 1
                    elif direction == "up" and prev_direction == "forward":
                        count_up += 1
                    elif direction == "down" and prev_direction == "forward":
                        count_down += 1

                prev_direction = direction

    cap.release()

    return [count_left, count_right, count_up, count_down]





def crop_eye(img, eye_points, IMG_SIZE):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

def get_video_time(video_path):
  
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  if fps >0:
      
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    time = total_frames/fps
  else:
      return
  return time

video_time = get_video_time('my_video.mp4') 
print(video_time) # Prints the total time in seconds

def blink_calculator(video_path):
  IMG_SIZE = (34, 26)
  blink_in_progress = False 
  blink_start_time = None
  blink_counter = 0
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('C:/Users/rayen/Downloads/Compressed/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')

  model = load_model('model/blink_detector.h5')
  cap = cv2.VideoCapture(video_path)

  while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
      break
 
    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
      shapes = predictor(gray, face)
      shapes = face_utils.shape_to_np(shapes)

      eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42], IMG_SIZE=IMG_SIZE)
      eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48], IMG_SIZE=IMG_SIZE)

      eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
      eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
      eye_img_r = cv2.flip(eye_img_r, flipCode=1)

      eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
      eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

      pred_l = model.predict(eye_input_l)
      pred_r = model.predict(eye_input_r)

      if pred_l < 0.1 and pred_r < 0.1 and not blink_in_progress:  
        blink_in_progress = True
        blink_start_time = time.time()  
    
      # If blink in progress, check if ended    
      elif blink_in_progress: 
        if pred_l > 0.1 and pred_r > 0.1:  
          blink_in_progress = False  
          blink_end_time = time.time()
          if blink_end_time - blink_start_time > 0.2:
            blink_counter += 1

  return blink_counter

def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq

def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared','sad']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return label

def normalize_values(points,disp):

    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    #print(stress_value)
    if stress_value>=75:
        return stress_value,"High Stress"
    else:
        return stress_value,"low_stress"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")    
emotion_classifier = load_model("model/_mini_XCEPTION.102-0.66.hdf5", compile=False)
points=[]
stress_value_list=[]

def measure_stress(video_path):
    cap = cv2.VideoCapture(video_path) 
    while cap.isOpened():

        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        #frame = imutils.resize(frame, width=500,height=500)
        if not _:
            break
        
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

        #preprocessing the image
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        detections = detector(gray,0)
        for detection in detections:
            emotion = emotion_finder(detection,gray)
            cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            shape = predictor(frame,detection)
            shape = face_utils.shape_to_np(shape)
            
            leyebrow = shape[lBegin:lEnd]
            reyebrow = shape[rBegin:rEnd]
                
            reyebrowhull = cv2.convexHull(reyebrow)
            leyebrowhull = cv2.convexHull(leyebrow)

            cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

            distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
            stress_value,stress_label = normalize_values(points,distq)
            cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            stress_value_list.append(stress_value)
    
        

    mean_stress = np.mean(stress_value_list)
    return mean_stress

def stress_analysis(blinks,stress):
    # Calculate the number of blinks in the video
    num_blinks = blinks
    
    # Measure the stress level in the video
    stress_level = stress
    
    # Analyze the relationship between the number of blinks and stress level
    if num_blinks > 0.5 and stress_level > 0.75:
        analysis = "The person in the video seems to be under high stress."
    elif num_blinks <= 0.5 and stress_level <= 0.75:
        analysis = "The person in the video seems to be relatively less stressed."
    else:
        analysis = "The stress level of the person in the video is not clear."
    
    return analysis