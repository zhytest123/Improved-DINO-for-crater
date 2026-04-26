import os
import torch
import json
import numpy as np
from main import build_model_main
from util.slconfig import SLConfig
from util import box_ops
from PIL import Image
import datasets.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# 路径配置
model_config_path = "ckpts/dinopp4/DINOpp_4scale.py"
model_checkpoint_path = "ckpts/dinopp4/checkpoint_best_regular.pth"
annotation_path = "./数据集筛选结果/danet数据集/high/annotations/instances_filtered.json"
image_dir = "./数据集筛选结果/danet数据集/high/images"

# 加载模型配置和检查点
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# 加载COCO类别名称
with open('util/crater_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k): v for k, v in id2name.items()}

# 图像预处理
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载COCO注释
coco = COCO(annotation_path)

# 遍历所有图片
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
threshold = 0.3
preds_coco_format = []

for img_file in tqdm(image_files, desc="Predicting"):
    image_path = os.path.join(image_dir, img_file)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    img_tensor, _ = transform(image, None)
    output = model.cuda()(img_tensor[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']

    # 获取图像ID
    image_id = None
    for img in coco.dataset['images']:
        if img['file_name'] == img_file:
            image_id = img['id']
            break
    if image_id is None:
        continue

    for i in range(len(scores)):
        if scores[i] > threshold:
            bbox = boxes[i].cpu().numpy().tolist()
            preds_coco_format.append({
                "image_id": image_id,
                "category_id": labels[i].item(),
                "bbox": [
                    bbox[0] * image_width,
                    bbox[1] * image_height,
                    (bbox[2] - bbox[0]) * image_width,
                    (bbox[3] - bbox[1]) * image_height
                ],
                "score": scores[i].item()
            })

# 保存预测结果
preds_path = 'preds_val2017.json'
with open(preds_path, 'w') as f:
    json.dump(preds_coco_format, f)

# COCO评估
coco_pred = coco.loadRes(preds_path)
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

os.remove(preds_path)