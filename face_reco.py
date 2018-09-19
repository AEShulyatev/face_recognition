import dlib
import cv2
from scipy.spatial import distance


def get_name(face_descriptor):
    name = "unknown"
    for i in range(len(base)):
        if distance.euclidean(face_descriptor, base[i]) < 0.6:
            name = names[i]
    return name


base = list()
names = ['Ivan', 'Victor', 'Artem', 'Grisha']

sp = dlib.shape_predictor('databases\\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('databases\\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
font = cv2.FONT_HERSHEY_SIMPLEX


def add_to_base(path):
    img = cv2.imread(path)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
    base.append(facerec.compute_face_descriptor(img, shape))


add_to_base("photos\\Ivan.jpg")
add_to_base("photos\\Victor.jpg")
add_to_base("photos\\Artem.jpg")
add_to_base("photos\\Grisha.jpg")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# win1 = dlib.image_window()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
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
