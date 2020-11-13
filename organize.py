import argparse, sys
import numpy as np
import face_recognition
import os
import time
from shutil import copyfile
from functools import reduce
from operator import or_
from collections import defaultdict
from PIL import Image
from tensorflow import keras

images = [
    'Sample images/11.jpg', 'Sample images/1585990613_wfh-memes-30.webp',
    'Sample images/919175-1-johnny-lever-birthday-memes.jpg',
    'Sample images/both1.jpg', 'Sample images/both2.jpg',
    'Sample images/both3.jpg', 'Sample images/handwritten (1).jpeg',
    'Sample images/handwritten (1).jpg', 'Sample images/handwritten (1).png',
    'Sample images/handwritten (2).jpeg', 'Sample images/handwritten (2).jpg',
    'Sample images/handwritten (2).png', 'Sample images/handwritten (3).jpeg',
    'Sample images/handwritten (3).jpg', 'Sample images/images.png',
    'Sample images/Modi1.jpg', 'Sample images/Modi2.jpg',
    'Sample images/Modi3.jpg', 'Sample images/pjimage-91-1594658167.jpg',
    'Sample images/resume (10).jpeg', 'Sample images/resume (10).jpg',
    'Sample images/resume (10).png', 'Sample images/resume (11).jpeg',
    'Sample images/resume (11).jpg', 'Sample images/resume (11).png',
    'Sample images/resume (12).jpeg', 'Sample images/resume (12).jpg',
    'Sample images/resume (12).png', 'Sample images/screenshot (14).jpeg',
    'Sample images/screenshot (2).png', 'Sample images/screenshot (39).png',
    'Sample images/screenshot (3).gif', 'Sample images/screenshot (3).png',
    'Sample images/screenshot (40).png', 'Sample images/screenshot (41).png',
    'Sample images/screenshot (42).png', 'Sample images/screenshot (43).png',
    'Sample images/screenshot (44).png', 'Sample images/screenshot (45).png',
    'Sample images/screenshot (46).png', 'Sample images/screenshot (47).png',
    'Sample images/screenshot (48).png', 'Sample images/screenshot (49).png',
    'Sample images/screenshot (4).gif', 'Sample images/screenshot (50).png',
    'Sample images/screenshot (51).png', 'Sample images/screenshot (52).jpg',
    'Sample images/screenshot (52).png', 'Sample images/screenshot (53).jpg',
    'Sample images/screenshot (53).png', 'Sample images/screenshot (54).jpg',
    'Sample images/screenshot (55).jpg', 'Sample images/screenshot (91).png',
    'Sample images/screenshot (92).jpeg', 'Sample images/screenshot (92).jpg',
    'Sample images/screenshot (92).png', 'Sample images/screenshot (93).jpeg',
    'Sample images/screenshot (93).jpg', 'Sample images/screenshot (93).png',
    'Sample images/Trunp_1.jpg', 'Sample images/Trunp_2.jpg',
    'Sample images/Trunp_3.jpg'
]


def predict_class(model, class_name, image_paths):
    """
    Run the model on the given img and return prediction
    Image is resized according to first layer in the model

    :param model: Image classification model
    :type model: keras.models.Sequential
    :param class_name: List of class names which will be used to decode model predictions
    :type class_name: list(str)
    :param img_path: List of image paths
    :type img_path: list(str)
    """
    _, *image_size, _ = model.layers[0].input_shape

    def preprocess(img_path):
        img = Image.open(img_path).convert("RGB").resize(image_size)
        return np.array(img, dtype=np.uint8)

    model_input = np.array([preprocess(img) for img in image_paths],
                           dtype=np.uint8)
    predictions = model.predict(model_input)
    for img, p in zip(image_paths, predictions):
        prediction = np.argmax(p)
        yield img, class_name[prediction], p[prediction]


def face_group(image_paths):
    face_map = {}
    for img in image_paths:
        list_of_face_encodings = face_recognition.face_encodings(
            face_recognition.load_image_file(img))
        if list_of_face_encodings == []: continue
        atleast_one_match = False
        for face in face_map:
            current_faces = np.array(list_of_face_encodings)
            past_face = np.frombuffer(face)
            if any(face_recognition.compare_faces(current_faces, past_face)):
                face_map[face].add(img)
                atleast_one_match = True
        if not atleast_one_match:
            for enc in list_of_face_encodings:
                encoded_str = np.array([enc]).tostring()
                face_map[encoded_str] = {img}
    return face_map.values()


def seperate(sets):
    """
    Given a list of sets, seperate them into exclusive groups

    :param sets: List of sets
    :type sets: list(set)
    """
    exclusive = defaultdict(set)
    for set_ in sets:
        for elem in set_:
            tmp = []
            for i, s in enumerate(sets, 1):
                if elem in s: tmp.append(i)
            tmp.sort()
            key = tuple(tmp)
            exclusive[key].add(elem)
    return exclusive


def classify(image_paths, face_threshold, classify_threshold):
    """
    Return a dictionary mapping category name to image paths
     - Documents
     - Memes
     - Screenshots
     - Unorganized
     - Face 1
     - Face 2
     - Face 1 & 2
    :param image_paths: Set of image paths
    :type image_paths: set(str)
    :param face_threshold: Minimum number of faces to make it its own category
    :type face_threshold: int
    :param classify_threshold: Minimum accuracy [0-1] to put image into category
    :type classify_threshold: float
    """
    start = time.perf_counter()
    dirs = defaultdict(set)
    faces = [
        imgs for imgs in face_group(image_paths) if len(imgs) >= face_threshold
    ]

    # Remove all images with face from set of images
    image_paths -= reduce(or_, faces)

    for face_idx, imgs in seperate(faces).items():
        folder_name = "Face " + " and ".join(map(str, face_idx))
        dirs[folder_name] = imgs

    end = time.perf_counter()
    print("Face recognition:", end - start)

    lookup = ['Documents', 'Memes', 'Screenshots']
    model = keras.models.load_model("image_organizer_modelv2.0.h5")

    for img, class_name, confidence in predict_class(model, lookup,
                                                     image_paths):
        if confidence >= classify_threshold:
            dirs[class_name].add(img)
        else:
            dirs["Unorganized"].add(img)
    
    end = time.perf_counter()
    print("Classification:",end-start)
    return dirs


classes = classify(set(images), 3, 0.60)
print(classes)

for subdir, imgs in classes.items():
    if os.path.exists(subdir):
        print("Directory exists!")
        break
    os.mkdir(subdir)
    for img in imgs:
        basename = os.path.basename(img)
        copyfile(img, os.path.join(subdir, basename))
