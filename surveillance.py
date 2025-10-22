import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import winsound

cam = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()

if not cam.isOpened():
    print("Erro ao abrir a camera")

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Erro ao ler a camera")
        break
    fg_mask = backSub.apply(frame)

    contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    min_contour_area = 500
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
    ]

    frame_ct = cv2.drawContours(frame, large_contours, -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame_ct)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
