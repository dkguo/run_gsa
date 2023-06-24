import itertools
import os
import re
from multiprocessing import Pool
from multiprocessing.connection import Client

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_tools.config import dataset_path
from dataset_tools.loaders import get_camera_names, get_num_frame


def extract_float_from_string(text):
    pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
    matches = re.findall(pattern, text)
    return float(matches[0][0]) if matches else None


def save_plt(image, prediction, save_path):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

    labels, masks, boxes = prediction
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)

    for box, label in zip(boxes, labels):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        plt.text(x0, y0, label)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def request_prediction(image, text_prompt):
    server_address = ('128.2.205.54', 60888)  # ('localhost', 5000)
    try:
        with Client(server_address) as conn:
            conn.send((image, text_prompt))
            prediction = conn.recv()
            return prediction
    except Exception as e:
        print(f'Error: {e}')
        return None


def predict(frame, camera_path, object_names):
    image_path = f'{camera_path}/rgb/{frame:06d}.png'
    save_dir = f'{camera_path}/object_pose/gsa'
    csv_path = f'{save_dir}/object_boxes.csv'

    text_prompt = "".join([f'{obj} . ' for obj in object_names])
    print(f'Predicting {image_path} with prompt: {text_prompt}')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = request_prediction(image, text_prompt)
    if prediction is None:
        return

    labels, masks, boxes = prediction
    save_plt(image, [labels, masks, boxes], f'{save_dir}/plot/{frame:06d}.jpg')
    for mask, label, box in zip(masks, labels, boxes):
        mask = mask[0].astype('uint8') * 255
        conf = extract_float_from_string(label)
        for object_name in object_names:
            if object_name in label:
                mask_path = f'{save_dir}/{object_name}/{frame:06d}.png'
                if os.path.exists(mask_path):
                    df = pd.read_csv(csv_path)
                    max_conf = df.loc[(df['frame'] == frame) & (df['object_name'] == object_name), 'conf'].max()
                    if conf < max_conf:
                        continue
                cv2.imwrite(mask_path, mask)
                with open(csv_path, 'a') as f:
                    f.write(f'{scene_name}, {camera_name}, {frame}, {object_name}, gsa, {conf}, "{box.tolist()}"\n')


if __name__ == '__main__':
    scene_name = 'scene_230313171600'
    num_predictor = 10
    object_names = ['bowl', 'hand']
    scene_path = f'{dataset_path}/{scene_name}'

    # create save directories
    for camera_name in get_camera_names(scene_path):
        camera_path = f'{scene_path}/{camera_name}'
        save_dir = f'{camera_path}/object_pose/gsa'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/plot', exist_ok=True)
        for object_name in object_names:
            os.makedirs(f'{save_dir}/{object_name}', exist_ok=True)
        with open(f'{save_dir}/object_boxes.csv', 'w') as f:
            f.write('scene_name, camera_name, frame, obj_name, predictor, confidence, box\n')

    with Pool(num_predictor) as pool:
        pool.starmap(predict,
                     list(itertools.product(
                         range(get_num_frame(scene_path)),
                         [f'{scene_path}/{camera_name}' for camera_name in get_camera_names(scene_path)],
                         [object_names]))
                     )


# if __name__ == '__main__':
#     import glob
#     import shutil
#
#     scene_name = 'scene_230313171600'
#     paths = glob.glob(f'{dataset_path}/{scene_name}/*/object_pose/gsa')
#     for path in paths:
#         shutil.rmtree(path)
