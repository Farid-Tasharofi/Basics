import numpy as np
import cv2
import os
from cv2 import aruco
import matplotlib.pyplot as plt
import requests


# create the board
workdir = "./workdir/"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(12, 9, 1, .8, aruco_dict)
imboard = board.draw((2000, 2000))
cv2.imwrite("./tiff/chessboard.tiff", imboard)

# get sample images
datadir = "C:/Users/Farid/Desktop/charuco/chArUco/data/calib_tel_ludo/check/void/"
images = np.array(
    [datadir + f for f in os.listdir(datadir) if f.endswith(".png")])
order = np.argsort([str(p.split(".")[-2].split("/")[-1]) for p in images])
images = images[order]


def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize


allCorners, allIds, imsize = read_chessboards(images)


def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[1000.,    0., imsize[0]/2.],
                                 [0., 1000., imsize[1]/2.],
                                 [0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


# Commented out IPython magic to ensure Python compatibility.


# CAMERA CALIBRATION
ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize)


# Use of camera calibration to estimate 3D translation and rotation of each marker on a scene
def grab_frame(cap):
    img_resp = requests.get("http://192.168.43.1:8080/photo.jpg")

    file = open(
        "C:/Users/Farid/Desktop/charuco/chArUco/data/calib_tel_ludo/response/sample_image.png", "wb")

    file.write(img_resp.content)

    file.close()

    frame = cv2.imread(
        "C:/Users/Farid/Desktop/charuco/chArUco/data/calib_tel_ludo/response/sample_image.png")
    frame = cv2.undistort(src=frame, cameraMatrix=mtx, distCoeffs=dist)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                          parameters=parameters)
    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize=(
            3, 3), zeroZone=(-1, -1), criteria=criteria)

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    # print(corners)

    plt.figure()
    plt.imshow(frame_markers, interpolation="nearest")
    plt.show()

    size_of_marker = 0.0285  # side lenght of the marker in meter
    rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(
        corners, size_of_marker, mtx, dist)

    length_of_axis = 0.1
    imaxis = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    for i in range(len(tvecs)):
        imaxis = aruco.drawAxis(
            imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)

    frame = np.array(imaxis, dtype="uint8")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Initiate the two cameras
cap1 = cv2.VideoCapture(0)

# create two subplots
ax1 = plt.subplot(1, 1, 1)

# create two image plots
im1 = ax1.imshow(grab_frame(cap1))

plt.ion()

while True:
    im1.set_data(grab_frame(cap1))
    plt.pause(30)
