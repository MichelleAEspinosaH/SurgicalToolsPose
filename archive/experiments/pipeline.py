## Imports
import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from ultralytics import YOLO

path_fullimage = "Clean/OutputImages/TestImageClean5.jpg"
path_objectimage = "Clean/OutputImages/TestImageClean52.jpg"

def save_frame(imagepath):
    ## Get frame from RGB camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)   

    ## Save image
    img = cv2.imwrite(imagepath, frame)
    return img

def detect_orb(imagepath):
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    cv2.imwrite("Clean/OutputImages/TestImageClean5_ORB.jpg", img2)
    return img2

def detect_sift(imagepath):
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp = sift.detect(img,None)
    kp, des = sift.compute(img, kp)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    cv2.imwrite("Clean/OutputImages/TestImageClean5_SIFT.jpg", img2)
    return img2

def detect_fast(imagepath):
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    cv2.imwrite("Clean/OutputImages/TestImageClean5_FAST.jpg", img2)
    return img2

def detect_YOLO(imagepath):
    model = YOLO("yolo26n-seg.pt")
    results = model.predict(imagepath)
    cv2.imwrite("Clean/OutputImages/TestImageClean5_YOLO.jpg", results[0].plot())
    return results[0].plot()

def query_SIFT(refpath, querypath):
    refimg = cv2.imread(refpath, 0)
    queryimg = cv2.imread(querypath, 0)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(refimg, None)
    kp2, des2 = sift.detectAndCompute(queryimg, None)

    # FLANN-based matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test to find good matches
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    img_matches = cv2.drawMatches(refimg, kp1, queryimg, kp2, good_matches, None)
    cv2.imwrite("Clean/OutputImages/TestImageClean5_QuerySIFT.jpg", img_matches)
    return img_matches

def query_ORB(refpath, querypath):
    import cv2

    # Load images (grayscale)
    refimg = cv2.imread(refpath, 0)
    queryimg = cv2.imread(querypath, 0)

    # Initialize ORB
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(refimg, None)
    kp2, des2 = orb.detectAndCompute(queryimg, None)

    # Safety check
    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return None

    # BFMatcher with Hamming distance (correct for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN match
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test (slightly looser than SIFT)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Draw matches
    img_matches = cv2.drawMatches(
        refimg, kp1,
        queryimg, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save result
    cv2.imwrite("Clean/OutputImages/TestImageClean5_QueryORB.jpg", img_matches)

    return img_matches

def query_FAST(refpath, querypath):
    import cv2

    # Load images
    refimg = cv2.imread(refpath, 0)
    queryimg = cv2.imread(querypath, 0)

    # FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints
    kp1 = fast.detect(refimg, None)
    kp2 = fast.detect(queryimg, None)

    # BRIEF descriptor extractor (requires opencv-contrib)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp1, des1 = brief.compute(refimg, kp1)
    kp2, des2 = brief.compute(queryimg, kp2)

    # Safety check
    if des1 is None or des2 is None:
        print("No descriptors found.")
        return None

    # BFMatcher with Hamming (binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Draw matches
    img_matches = cv2.drawMatches(
        refimg, kp1,
        queryimg, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite("Clean/OutputImages/TestImageClean5_QueryFAST.jpg", img_matches)

    return img_matches

#save_frame(path_fullimage) 

detect_YOLO(path_fullimage)
detect_fast(path_fullimage)
detect_orb(path_fullimage)
detect_sift(path_fullimage)

query_FAST(path_objectimage, path_fullimage)
query_ORB(path_objectimage, path_fullimage)
query_SIFT(path_objectimage,path_fullimage)

