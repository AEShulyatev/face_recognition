import os
import dlib
import cv2
from scipy.spatial import distance
from scipy.misc import imresize
from time import clock

names, base = eval(open('names_descriptors.txt').read())

sp = dlib.shape_predictor('datasets\\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('datasets\\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
font = cv2.FONT_HERSHEY_SIMPLEX


def get_name(face_descriptor):
    name = "unknown"
    min_ = 2
    for i in range(len(base)):
        dist = distance.euclidean(face_descriptor, base[i])
        if dist < 0.6 and dist < min_:
            min_ = dist
            name = names[i]
    return name


print("[INFO] starting video stream...")
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\Фильмы\Матрица.mkv")
cap.set(3, 640)
cap.set(4, 480)
start = clock()
print(start)
# win1 = dlib.image_window()

while True:
    cap.set(cv2.CAP_PROP_POS_MSEC, (clock() - start) * 1000 + 500000)
    ret, img = cap.read()
    if ret:
        # img = cv2.flip(img, 1)
        # img = imresize(img, (img.shape[0] * 2, img.shape[1] * 2))
        # win1.set_image(img)
        dets = detector(img, 0)
        for i in range(len(dets)):
            d = dets[i]

            shape = sp(img, d)
            # win1.add_overlay(d)
            # win1.add_overlay(shape)
            name = get_name(facerec.compute_face_descriptor(img, shape))
            cv2.putText(img, name, (d.left() + 5, d.bottom() - 5), font, 1,
                        (255, 255, 255), 2)

            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Name: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom(), name))

            cv2.rectangle(img, (d.left(), d.bottom()), (d.right(), d.top()), (255, 0, 0), 2)
        cv2.imshow('video', img)
        cv2.waitKey(1)
    # win1.clear_overlay()
