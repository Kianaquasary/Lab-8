import cv2
import numpy as np

# We load reference image
reference_img = cv2.imread('ref-point.jpg', 0)

# We set up camera capture
cap = cv2.VideoCapture(0)

# Now we should use SIFT detector and matcher
sift = cv2.SIFT_create()
matcher = cv2.FlannBasedMatcher()

# here we detect keypoints and descriptors from reference image
kp1, des1 = sift.detectAndCompute(reference_img, None)

while True:

    # We read a frame from the camera
    ret, frame = cap.read()

    # Now we should convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Now we detect keypoints and descriptors for current frame
    kp2, des2 = sift.detectAndCompute(gray, None)

    # Here we match keypoints from reference image to current frame
    matches = matcher.knnMatch(des1, des2, k=2)

    # In this part we use filter that matches based on distance ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Drawing matches on current frame
    img_matches = cv2.drawMatches(reference_img, kp1, gray, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Now in the end we check if enough good matches were found in camera
    if len(good_matches) > 10:
        print("Reference image found!")
        cv2.imshow("Reference Image", reference_img)
    else:
        print("Reference image NOT found!")
        cv2.imshow("Matching Result", img_matches)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

