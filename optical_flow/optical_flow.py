import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


# Read in the images
frame_1 = cv2.imread('images/pacman_1.png')
frame_2 = cv2.imread('images/pacman_2.png')
frame_3 = cv2.imread('images/pacman_3.png')

# convert to RGB
frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2RGB)


# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('frame 1')
ax1.imshow(frame_1)
ax2.set_title('frame 2')
ax2.imshow(frame_2)
ax3.set_title('frame 3')
ax3.imshow(frame_3)
plt.show()


# parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.2,
                       minDistance = 5,
                       blockSize = 5 )


# convert all frames to grayscale
gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
gray_3 = cv2.cvtColor(frame_3, cv2.COLOR_RGB2GRAY)

images = [gray_1, gray_2, gray_3]

for i, img in enumerate(images):

    if i == len(images): break

    new_img = images[i+1]

    ### KEYPOINT DETECTION ###

    # Take first frame and find corner points in it
    pts = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)

    # display the detected points
    plt.imshow(img)
    for p in pts:
        # plot x and y detected points
        plt.plot(p[0][0], p[0][1], 'r.', markersize=15)
    plt.show()
    # print out the x-y locations of the detected points
    print("Frame {} has points: {}".format(i, pts))


    ### OPTICAL FLOW ###

    # parameters for lucas kanade optical flow
    lk_params = dict(winSize=(5, 5),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # calculate optical flow between first and second frame
    pts_2, match, err = cv2.calcOpticalFlowPyrLK(img, new_img, pts, None, **lk_params)

    # Select good matching points between the two image frames
    good_new = pts_2[match == 1]
    good_old = pts[match == 1]

    # create a mask image for drawing (u,v) vectors on top of the second frame
    mask = np.zeros_like(new_img)

    # draw the lines between the matching points (these lines indicate motion vectors)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # draw points on the mask image
        mask = cv2.circle(mask, (a, b), 5, (200), -1)
        # draw motion vector as lines on the mask image
        mask = cv2.line(mask, (a, b), (c, d), (200), 3)
        # add the line image and second frame together

    composite_im = np.copy(new_img)
    composite_im[mask!=0] = [0]


    plt.imshow(composite_im)
    plt.show()

