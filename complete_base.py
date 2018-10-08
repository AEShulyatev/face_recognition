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


def add_to_base(path):
    img = cv2.imread(path)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
    base.append(facerec.compute_face_descriptor(img, shape))


def complete_base():
    len_ = len(os.listdir('photos'))
    for i, f in enumerate(os.listdir('photos')):
        print("[INFO] processing image {}/{}".format(i + 1, len_))
        names.append(f[:-4])
        add_to_base('photos\\' + f)


complete_base()

with open('names_descriptors.txt', 'w') as f:
    f.write(repr([names, base]))
