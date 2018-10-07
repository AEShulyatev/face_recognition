import cv2
import dlib
from scipy.misc import imresize

sp = dlib.shape_predictor('datasets\\shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
win1 = dlib.image_window()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = imresize(img, (img.shape[0] * 2, img.shape[1] * 2))
    win1.set_image(img)
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)
        win1.add_overlay(d)
        win1.add_overlay(shape)
    cv2.waitKey(100)
    win1.clear_overlay()
