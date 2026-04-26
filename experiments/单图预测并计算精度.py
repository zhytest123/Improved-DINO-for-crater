import os
import torch
import json
import numpy as np
from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
from PIL import Image
import datasets.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 路径配置
model_config_path = "ckpts/dinopp4/DINOpp_4scale.py"  # 模型配置文件路径
model_checkpoint_path = "ckpts/dinopp4/checkpoint_best_regular.pth"  # 模型检查点路径
annotation_path = "./验证数据集/annotations/instances_filtered.json"  # COCO注释文件路径
# image_path = "./lronac数据集/val2017/-11.258656700010981,-9.88677586017368,-37.58491810486972,-36.213037265032405.png"  # 测试图像路径
# 对比：-162.16554908211486,-160.7936682422776,-0.5441354292623967,0.8277454105749196
# 对比：165.71397163900173,167.0858524788391,-12.89106298779817,-11.519182147960857
# 消融：172.57337583818827,173.94525667802563,-18.378586347147394,-17.006705507310105
# 消融：-63.39012861382869,-62.01824777399138,-19.75046718698471,-18.378586347147394
image_path = "./验证数据集/images/-162.16554908211486,-160.7936682422776,-0.5441354292623967,0.8277454105749196.png"  # 自定义测试图像路径

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

# 加载图像
image = Image.open(image_path).convert("RGB")
# 获取图像尺寸
image_width, image_height = image.size

# 图像预处理
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image, _ = transform(image, None)
print("Image shape after preprocessing:", image.shape)

# 图像预测
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

# 可视化预测结果
threshold = 0.3  # 设置阈值
vslzr = COCOVisualizer()

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > threshold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}
#vslzr.visualize(image, pred_dict, savedir=None, dpi=150)

# 加载COCO注释
coco = COCO(annotation_path)

# 从COCO注释中获取图像ID
image_id = None
for img in coco.dataset['images']:
    if img['file_name'] == os.path.basename(image_path):
        image_id = img['id']
        break

if image_id is None:
    raise ValueError("在COCO注释中未找到图像")

# 获取图像的真实标注
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)

# 获取真实标注框
true_boxes = [ann['bbox'] for ann in anns]
true_labels = [id2name[ann['category_id']] for ann in anns]

# 转换真实标注框格式
true_boxes_cxcywh = []
for bbox in true_boxes:
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    true_boxes_cxcywh.append([cx / image_width, cy / image_height, w / image_width, h / image_height])

# 创建真实标注字典
true_dict = {
    'boxes': torch.Tensor(true_boxes_cxcywh),
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': true_labels
}

# 可视化真实标注框
vslzr.visualize(image, [(pred_dict, 'green', 'solid'), (true_dict, 'red', 'dashed')], savedir=None, dpi=150)
# 只显示预测结果
# vslzr.visualize(image, [(pred_dict, 'random', 'solid')], savedir=None, dpi=150)

# 将预测结果转换为COCO格式
preds_coco_format = []
for i in range(len(scores)):
    if scores[i] > threshold:
        bbox = output['boxes'][i].cpu().numpy().tolist()
        preds_coco_format.append({
            "image_id": image_id,
            "category_id": labels[i].item(),
            "bbox" : [bbox[0] * image_width, bbox[1] * image_height, (bbox[2] - bbox[0]) * image_width, (bbox[3] - bbox[1]) * image_height],
            #"bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # xyxy 转 xywh
            "score": scores[i].item()
        })
#print(preds_coco_format)

# 将预测结果保存到临时JSON文件
preds_path = 'preds.json'
with open(preds_path, 'w') as f:
    json.dump(preds_coco_format, f)

# 加载预测结果并进行评估
coco_pred = coco.loadRes(preds_path)
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = [image_id]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 清理临时文件
os.remove(preds_path)
