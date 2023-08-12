#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrFeatureExtractor
import json

# Custom class
class Detr(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", 
            num_labels = 43, 
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)


# Helper functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([size[0], size[1], size[0], size[1]], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=1))
        text = f'{p.argmax().item()}: {p[p.argmax()]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def visualize_predictions(image, outputs, threshold=0.9):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    plot_results(image, probas[keep], bboxes_scaled)
    return bboxes_scaled.tolist(), probas[keep].argmax(-1).tolist()

def get_model_path(sub_count, join_list):
    path = os.path.dirname(os.path.realpath(__file__))
    for _ in range(sub_count):
        path = os.path.dirname(os.path.normpath(path))
    for directory in join_list:
        path = os.path.join(path, directory)
    return path


def main(args):
    model_path = get_model_path(1, ['model', 'model.ckpt'])
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    id_to_name = {category['id']: category['name'] for category in data['categories']}
    im = Image.open(args.img_path)
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = Detr()
    model = model.load_from_checkpoint(model_path)
    model.eval()
    encoding = feature_extractor(im, return_tensors="pt")
    outputs = model(**encoding)
  
    bboxes_displayed, classes_displayed = visualize_predictions(im, outputs)
    classes_displayed = [id_to_name[id] for id in classes_displayed]
    new_bboxes = [[x, y, width-x, height-y] for x, y, width, height in bboxes_displayed]

    log_output = {
        "bboxes": new_bboxes,
        "components": classes_displayed
    }
    json_string = json.dumps(log_output)
    with open(args.uuid + ".json", 'w') as outfile:
        outfile.write(json_string)
    print(log_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img_path", required=True)
    parser.add_argument("-j", "--json_path", required=True, help="Path to JSON file with class names")
    parser.add_argument("-u", "--uuid", required=True, help="uuid to write JSON file with class names and bboxes")
    args = parser.parse_args()
    main(args)

