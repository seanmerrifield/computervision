import torch
import cv2
import numpy as np
from pathlib import Path

from model import Net

# Load Trained Model
MODEL_PATH = str(Path('train', 'run_17', 'trained_model.pt'))
assert Path(MODEL_PATH).exists(), 'Path to trained model does not exist.'

net = Net()
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()


def detect_keypoints(image, x, y, w, h):
    # Select the region of interest that is the face in the image
    p = 65  #Extra padding around region of interest
    roi = image[y - p:y + h + p, x - p:x + w + p]
    cv2.rectangle(roi, (0, 0), (roi.shape[0], roi.shape[1]), (255, 0, 0), 2)

    ## Convert the face region from RGB to grayscale
    img_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    if img_gray == None: return

    ## Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    img_gray = img_gray / 255.0

    ## Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    img = cv2.resize(img_gray, (96, 96))

    ## Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    img_tensor = torch.tensor(img)
    img_tensor = img_tensor.float()

    ## Make facial keypoint predictions using loaded, trained network
    # forward pass to get net output
    # Convert tensor from double to float
    output_pts = net(img_tensor)

    # reshape to batch_size x 68 x 2 pts
    output_pts = output_pts.view(68, 2)
    output_pts = output_pts.data.numpy()

    # undo normalization of keypoints
    output_pts = output_pts * 50 + 100

    output_pts[:, 0] *= roi.shape[0] / 96
    output_pts[:, 1] *= roi.shape[1] / 96

    for pt_x, pt_y in output_pts:
        cv2.circle(roi, (int(pt_x), int(pt_y)),  2, (0,255,0), -1)

    return output_pts

cascPath = "./detector_architectures/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        detect_keypoints(frame, x, y, w, h)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()