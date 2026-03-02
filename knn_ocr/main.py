import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread


def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 0
    lb = label(binary)
    props = max(regionprops(lb), key=lambda r: r.area)
    return np.array([*props.moments_hu, props.eccentricity], dtype='f4')


def make_train(path):
    train = []
    responses = []
    ncls = 0
    for cls in sorted(path.glob("*")):
        ncls += 1
        for p in cls.glob("*.png"):
            train.append(extractor(imread(p)))
            responses.append(ncls)
    train = np.array(train, dtype="f4").reshape(-1, 8)
    responses = np.array(responses, dtype="f4").reshape(-1, 1)
    return train, responses

my_str = input("Enter path to files: ")

data = Path(my_str)

cloud = ['+', '-', 'A', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'N', 'O', 'P', 'R', 'S', 'T',
         'U', 'V', 'W', 'Y', 'a', 'c', 'h', 'i', 'k', 'n', 'o', 'p', 'r', 's', 't', 'u',
         'v', 'y']

cnt = 1
for i in range(7):
    image = imread(data / f"{i}.png")
    train, responses = make_train(data / "train")
    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)

    gray = image.mean(2)
    binary = gray > 0
    lb = label(binary)
    props = regionprops(lb)

    sorted_props = sorted(props, key=lambda prop: prop.centroid[1])

    find = []
    prev_max_col = None
    space_indices = []
    char_index = 0

    for prop in range(len(sorted_props)):
        if sorted_props[prop].area < 300:
            continue

        _, min_col, _, max_col = sorted_props[prop].bbox

        if prev_max_col is not None:
            if min_col - prev_max_col > 25:
                space_indices.append(char_index)

        find.append(extractor(sorted_props[prop].image))
        prev_max_col = max_col
        char_index += 1


    find = np.array(find, dtype="f4").reshape(-1, 8)
    ret, results, neighbours, dist = knn.findNearest(find, 5)

    message = ''
    for idx, result in enumerate(results.flatten()):
        if idx in space_indices:
            message += " "
        message += cloud[int(result) - 1]

    print(f"На картинке №{cnt} сообщение: {message}")
    cnt += 1


