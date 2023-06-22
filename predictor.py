from multiprocessing import Process
from multiprocessing.connection import Listener, Client

import torch
import torchvision

import sys

from midman import midman_address

gsa_path = '/home/jianrenw/project_data/Grounded-Segment-Anything'

sys.path.append(gsa_path)

from segment_anything import build_sam, SamPredictor
from utils import load_model, get_grounding_output, load_image_from_cv


class Predictor:
    def __init__(self, name):
        self.name = name

        checkpoint = f'{gsa_path}/sam_vit_h_4b8939.pth'
        print('Loading SAM predictor...')
        self.predictor = SamPredictor(build_sam(checkpoint=checkpoint))

        grounded_checkpoint = f'{gsa_path}/groundingdino_swint_ogc.pth'
        config_file = f'{gsa_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.device = 'cuda'
        print('Loading GroundingDINO model...')
        self.model = load_model(config_file, grounded_checkpoint, device=self.device)

    def predict(self, image, text_prompt, box_threshold=0.5, text_threshold=0.3, iou_threshold=0.5):
        print('Predicting grounding...')
        image_pil, image_transfromed = load_image_from_cv(image)
        print(image_pil.size, image_transfromed.size)
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.model, image_transfromed, text_prompt, box_threshold, text_threshold, device=self.device
        )

        print('Predicting segmentation...')
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        self.predictor.set_image(image)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        print('Prediction done.')
        return pred_phrases, masks.numpy(), boxes_filt.numpy()


def start_predictor():
    conn = Client(midman_address)
    conn.send('This is predictor.')
    name = conn.recv()
    predictor = Predictor(name)
    print(f'Predictor {name} connected to the server.')
    while True:
        try:
            args = conn.recv()
            print('Received arguments. Predicting...')
            prediction = predictor.predict(*args)
            conn.send(prediction)
        except:
            conn.send(None)


if __name__ == '__main__':
    start_predictor()
