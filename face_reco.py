import os
import dlib
import cv2
from scipy.spatial import distance
from scipy.misc import imresize

names = []
base = []

sp = dlib.shape_predictor('datasets\\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('datasets\\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
font = cv2.FONT_HERSHEY_SIMPLEX


def add_to_base(path):
    img = cv2.imread(path)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
    base.append(facerec.compute_face_descriptor(img, shape))


def complete_base():
    for f in os.listdir('photos'):
        names.append(f[:-4])
        add_to_base('photos\\' + f)


complete_base()


def get_name(face_descriptor):
    name = "unknown"
    min_ = 2
    for i in range(len(base)):
        dist = distance.euclidean(face_descriptor, base[i])
        if dist < 0.6 and dist < min_:
            min_ = dist
            name = names[i]
    return name


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# win1 = dlib.image_window()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = imresize(img, (img.shape[0] * 2, img.shape[1] * 2))
    # win1.set_image(img)
    dets = detector(img, 0)
    for i in range(len(dets)):
        d = dets[i]
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)
        # win1.add_overlay(d)
        # win1.add_overlay(shape)
        cv2.putText(img, get_name(facerec.compute_face_descriptor(img, shape)), (d.left() + 5, d.bottom() - 5), font, 1,
                    (255, 255, 255), 2)

        cv2.rectangle(img, (d.left(), d.bottom()), (d.right(), d.top()), (255, 0, 0), 2)
    cv2.imshow('video', img)
    cv2.waitKey(1)
    # win1.clear_overlay()
