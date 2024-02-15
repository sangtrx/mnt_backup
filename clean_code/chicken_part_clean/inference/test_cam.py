import cv2

# Camera index 0 (first camera)
cap0 = cv2.VideoCapture(0)

# Camera index 1 (second camera)
cap1 = cv2.VideoCapture(1)

# Camera index 2 (third camera)
cap2 = cv2.VideoCapture(2)

# Check if the cameras opened successfully
if not cap0.isOpened():
    print("Camera 0 not opened")
if not cap1.isOpened():
    print("Camera 1 not opened")
if not cap2.isOpened():
    print("Camera 2 not opened")

# Read frames from the cameras
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret0 or not ret1 or not ret2:
        print("Error reading frames from one or more cameras")
        break

    # Process frames or display them as needed
    # For example, you can display frames like this:
    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera objects and close windows
cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
