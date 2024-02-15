import cv2
import numpy as np

def detect_squares(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite('edges.jpg', edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
            squares.append((x, y, x+w, y+h))
    return squares

# Read an image
img = cv2.imread('/mnt/tqsang/hand_dataset_YOLO/datasets/val/images/part2_00008169.png')

# Detect squares
squares = detect_squares(img)

# Draw squares
for (x1, y1, x2, y2) in squares:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the image with detected squares
cv2.imwrite('squares_detected.jpg', img)
