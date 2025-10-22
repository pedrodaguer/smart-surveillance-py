import time
import cv2
import winsound

cam = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()
last_beep = 0
delay_frames = 1

if not cam.isOpened():
    print("Erro ao abrir a camera")

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Erro ao ler a camera")
        break
    fg_mask = backSub.apply(frame)
    delay_frames -=1
    contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    min_contour_area = 2500
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
    ]

    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)

    if large_contours and (time.time() - last_beep > 0.75) and delay_frames <= 0:
        winsound.Beep(1000, 75)
        last_beep = time.time()


    cv2.imshow('Live Feed', frame_out)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
