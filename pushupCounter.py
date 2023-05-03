import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio, math
from Helpers.helpers import *
from Helpers.helpers import _keypoints_and_edges_for_display
import pyttsx3

engine = pyttsx3.init()

# Load Model
model_name = "movenet_lightning_f16.tflite" #@param ["movenet_lightning", "movenet_thunder", "movenet_lightning_f16.tflite", "movenet_thunder_f16.tflite", "movenet_lightning_int8.tflite", "movenet_thunder_int8.tflite"]

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def _angle(line1, line2):
    """
    Calculate the angle between the left shoulder and left elbow
    
    Args:
        line1: A numpy array representing the left shoulder coordinates of shape (2, 2)
        line2: A numpy array representing the left elbow coordinates of shape (2, 2)

    Returns:
        The angle between the two lines in degrees
    """
    # Calculate the angle between the two lines
    x1 = line1[0][0]
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    a1 = line2[0][0]
    b1 = line2[0][1]
    a2 = line2[1][0]
    b2 = line2[1][1]

    angle = abs(np.arctan(abs((b2 - b1) / (a2 - a1)) + abs((y2 - y1) / (x2 - x1)))) * 180 / np.pi
    """
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (b2 - b1) / (a2 - a1)

    angle = math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2)))

    angle = math.degrees(angle)
    return 180 - angle
    """
    return angle    

count = 0
is_down_position = False
started = False
last_pushup_frame = 0
frame_number = 0
def determine_pushup(angle, prior_angles, threshold):
    global count
    global is_down_position
    global started
    global last_pushup_frame
    """
    Determine if the user has done one pushup position
    """
    if angle <= threshold and not is_down_position:
        is_down_position = True

    if angle > threshold and is_down_position:
        if count == 0:
            started = True
        is_down_position = False
        count += 1
        last_pushup_frame = frame_number
        # Text to speech
        engine.say(str(count))
        engine.runAndWait()

# Opencv stream
cap = cv2.VideoCapture(0)

num_frames, image_height, image_width, _ = (1, 480, 640, 3)
crop_region = init_crop_region(image_height, image_width)

angles = []
threshold = 65 # TODO: need a better way to determine when a pushup is done
num_right_elbow_detected = 0
num_left_elbow_detected = 0
num_elbow_detected_threshold = 5

prior_last_pushup_frame_threshold = 300

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    keypoints_with_scores = run_inference(
      movenet, frame, crop_region,
      crop_size=[input_size, input_size])

    (keypoint_locs, keypoint_edges,
        edge_colors, edge_names) = _keypoints_and_edges_for_display(
        keypoints_with_scores, image_height, image_width)
    
    #p = draw_prediction_on_image(
    #    frame.astype(np.int32),
    #    keypoints_with_scores, crop_region=None,
    #    close_figure=True, output_image_height=300)
    #cv2.imshow('frame', p)

    # Get index of left shoulder and left elbow
    left_shoulder_idx = edge_names.index('leftshoulder-leftelbow') if 'leftshoulder-leftelbow' in edge_names else None
    left_elbow_idx = edge_names.index('leftelbow-leftwrist') if 'leftelbow-leftwrist' in edge_names else None

    # Get the index of right shoulder and right elbow
    right_shoulder_idx = edge_names.index('rightshoulder-rightelbow') if 'rightshoulder-rightelbow' in edge_names else None
    right_elbow_idx = edge_names.index('rightelbow-rightwrist') if 'rightelbow-rightwrist' in edge_names else None
    # If left shoulder and left elbow are detected
    if left_shoulder_idx is not None and left_elbow_idx is not None:
        num_left_elbow_detected += 1

        if num_left_elbow_detected > num_elbow_detected_threshold:
            # Get the line from the left shoulder to the left elbow
            left_shoulder_line = keypoint_edges[left_shoulder_idx]
            # Get the line from the left elbow to the left wrist
            left_elbow_line = keypoint_edges[left_elbow_idx]

            angle = _angle(left_shoulder_line, left_elbow_line)

            determine_pushup(angle, angles, threshold)

            angles.append(angle)
        
    # If right shoulder and right elbow are detected
    elif right_shoulder_idx is not None and right_elbow_idx is not None:
        num_right_elbow_detected += 1

        if num_right_elbow_detected > num_elbow_detected_threshold:
            # Get the line from the right shoulder to the right elbow
            right_shoulder_line = keypoint_edges[right_shoulder_idx]
            # Get the line from the right elbow to the right wrist
            right_elbow_line = keypoint_edges[right_elbow_idx]

            angle = _angle(right_shoulder_line, right_elbow_line)

            determine_pushup(angle, angles, threshold)

            angles.append(angle)
    else:
        if num_left_elbow_detected > 0:
            num_left_elbow_detected -= 1
        if num_right_elbow_detected > 0:
            num_right_elbow_detected -= 1

    frame_number += 1

    if started and frame_number - last_pushup_frame > prior_last_pushup_frame_threshold:
        started = False
        count = 0
        engine.say("That was pathetic. My grandma can do better than that.")
        engine.runAndWait()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Count:", count)
# Plot the angles
plt.plot(angles)
plt.show()

"""
image_path = 'pushupTest.gif'
image = tf.io.read_file(image_path)
image = tf.image.decode_gif(image)
# Load the input image.
num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

angles = []
is_down_position = False
threshold = 50
count = 0

for frame_idx in range(num_frames):
    keypoints_with_scores = run_inference(
      movenet, image[frame_idx, :, :, :], crop_region,
      crop_size=[input_size, input_size])

    (keypoint_locs, keypoint_edges,
        edge_colors, edge_names) = _keypoints_and_edges_for_display(
        keypoints_with_scores, image_height, image_width)

    # Get index of left shoulder and left elbow
    left_shoulder_idx = edge_names.index('leftshoulder-leftelbow') if 'leftshoulder-leftelbow' in edge_names else None
    left_elbow_idx = edge_names.index('leftelbow-leftwrist') if 'leftelbow-leftwrist' in edge_names else None

    # Get the index of right shoulder and right elbow
    right_shoulder_idx = edge_names.index('rightshoulder-rightelbow') if 'rightshoulder-rightelbow' in edge_names else None
    right_elbow_idx = edge_names.index('rightelbow-rightwrist') if 'rightelbow-rightwrist' in edge_names else None
    
    # If left shoulder and left elbow are detected
    if left_shoulder_idx is not None and left_elbow_idx is not None:
        # Get the line from the left shoulder to the left elbow
        left_shoulder_line = keypoint_edges[left_shoulder_idx]
        # Get the line from the left elbow to the left wrist
        left_elbow_line = keypoint_edges[left_elbow_idx]

        angle = _angle(left_shoulder_line, left_elbow_line)
        angles.append(angle)
    # If right shoulder and right elbow are detected
    elif right_shoulder_idx is not None and right_elbow_idx is not None:
        # Get the line from the right shoulder to the right elbow
        right_shoulder_line = keypoint_edges[right_shoulder_idx]
        # Get the line from the right elbow to the right wrist
        right_elbow_line = keypoint_edges[right_elbow_idx]

        angle = _angle(right_shoulder_line, right_elbow_line)
        angles.append(angle)

    if angle <= threshold and not is_down_position:
        is_down_position = True

    if angle > threshold and is_down_position:
        is_down_position = False
        count += 1

print("Count:", count)
# Plot the angles
plt.plot(angles)
plt.show()
"""