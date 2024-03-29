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
from dataset_tools.record.step_1_record_multiple_d435 import D435s
from dataset_tools.utils.name import get_camera_names, get_newest_scene_name


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
        label = label.replace(" _ ", "_")
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        plt.text(x0, y0, label)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def request_prediction(image, text_prompt, predictor='gsam'):
    if predictor == 'yolo':
        server_address = ('localhost', 6000)
    elif predictor == 'gsam':
        server_address = ('localhost', 60888)
    else:
        print(f'Unknown predictor: {predictor}')
        return None

    try:
        with Client(server_address) as conn:
            conn.send((image, text_prompt, 0.2, 0.2, 0.5))  # box_threshold, text_threshold, iou_threshold
            prediction = conn.recv()
            if prediction is None:
                print(f'Error: prediction is None')
            return prediction
    except Exception as e:
        print(f'Error: {e}')
        return None


def create_directories(scene_path, object_names):
    for camera_name in get_camera_names(scene_path):
        camera_path = f'{scene_path}/{camera_name}'
        save_dir = f'{camera_path}/masks'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/plot', exist_ok=True)
        for object_name in object_names:
            os.makedirs(f'{save_dir}/{object_name}', exist_ok=True)
        if not os.path.exists(f'{save_dir}/object_boxes.csv'):
            with open(f'{save_dir}/object_boxes.csv', 'w') as f:
                f.write('scene_name,camera_name,frame,object_name,predictor,confidence,box\n')


def predict_single_frame(object_names, scene_name, camera_name, frame, overwrite=False, predictor='gsam',
                         max_mask_precentage=0.15, max_box_precentage=0.3,
                         merge_all_masks=False, select_mask_manually=False):
    camera_path = f'{dataset_path}/{scene_name}/{camera_name}'
    image_path = f'{camera_path}/rgb/{frame:06d}.png'
    save_dir = f'{camera_path}/masks'
    csv_path = f'{save_dir}/object_boxes.csv'

    if not overwrite and os.path.exists(f'{save_dir}/plot/{frame:06d}.jpg'):
        return

    text_prompt = "".join([f'{obj} . ' for obj in object_names])
    print(f'Predicting {image_path} with prompt: {text_prompt}')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = request_prediction(image, text_prompt, predictor)
    if prediction is None:
        return

    labels, masks, boxes = prediction
    save_plt(image, [labels, masks, boxes], f'{save_dir}/plot/{frame:06d}.jpg')
    for mask, label, box in zip(masks, labels, boxes):
        mask = mask[0].astype('uint8') * 255
        if (mask.sum() / 255 > max_mask_precentage * mask.shape[0] * mask.shape[1] or
                box[2] - box[0] > max_box_precentage * mask.shape[1] or
                box[3] - box[1] > max_box_precentage * mask.shape[0]):
            continue
        conf = extract_float_from_string(label)
        label = label.replace(" _ ", "_")
        for object_name in object_names:
            if object_name in label:
                mask_path = f'{save_dir}/{object_name}/{frame:06d}.png'
                if select_mask_manually:
                    cv2.imshow('mask', cv2.addWeighted(image, 0.5,
                                                       cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR), 0.5, 0))
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if key != ord('s'):
                        continue
                    if os.path.exists(mask_path):
                        prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = np.maximum(mask, prev_mask).astype('uint8')

                elif os.path.exists(mask_path):
                    if merge_all_masks:
                        prev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = np.maximum(mask, prev_mask).astype('uint8')
                    else:
                        df = pd.read_csv(csv_path)
                        max_conf = df.loc[(df['frame'] == frame) & (df['object_name'] == object_name), 'confidence'].max()
                        if conf < max_conf:
                            continue
                cv2.imwrite(mask_path, mask)
                with open(csv_path, 'a') as f:
                    f.write(f'{scene_name}, {camera_name}, {frame}, {object_name}, gsa, {conf}, "{box.tolist()}"\n')


def predict_scene(scene_name, object_names, frame_nums, overwrite=False, num_predictor=10, predictor='gsam',
                  max_mask_precentage=0.15, max_box_precentage=0.3,
                  merge_all_masks=False, select_mask_manually=False):
    scene_path = f'{dataset_path}/{scene_name}'
    create_directories(scene_path, object_names)

    if select_mask_manually:
        for camera_name in get_camera_names(scene_path):
            predict_single_frame(object_names, scene_name, camera_name, frame_nums[0], overwrite, predictor,
                                 max_mask_precentage, max_box_precentage, merge_all_masks, select_mask_manually)
    else:
        with Pool(num_predictor) as pool:
            pool.starmap(predict_single_frame,
                         list(itertools.product(
                             [object_names], [scene_name], get_camera_names(scene_path),
                             frame_nums, [overwrite], [predictor], [max_mask_precentage], [max_box_precentage],
                             [merge_all_masks], [select_mask_manually]))
                         )


if __name__ == '__main__':
    scene_name = 'scene_231116075559_blue_cup'
    # cameras = D435s(scene_name, init_recorder=False)
    # frame_num = cameras.capture()
    object_names = ['hand']

    scene_path = f'{dataset_path}/{scene_name}'
    create_directories(scene_path, object_names)

    predict_scene(scene_name, object_names, [65], overwrite=True, num_predictor=20, select_mask_manually=True)


# if __name__ == '__main__':
#     import glob
#     import shutil
#
#     scene_name = 'scene_230704142825'
#     paths = glob.glob(f'{dataset_path}/{scene_name}/*/masks')
#     for path in paths:
#         shutil.rmtree(path)
