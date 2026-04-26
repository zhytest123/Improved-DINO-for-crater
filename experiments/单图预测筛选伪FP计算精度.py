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
annotation_path = "./数据集筛选结果/nac数据集/high/annotations/instances_filtered.json"  # COCO注释文件路径
image_path = "./数据集筛选结果/nac数据集/high/images/A14-3821.png"  # 自定义测试图像路径

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
vslzr.visualize(image, [(pred_dict, 'random', 'solid'), (true_dict, 'red', 'dashed')], savedir=None, dpi=150)

# 将预测结果转换为COCO格式
preds_coco_format = []
for i in range(len(scores)):
    if scores[i] > threshold:
        bbox = output['boxes'][i].cpu().numpy().tolist()
        preds_coco_format.append({
            "image_id": image_id,
            "category_id": labels[i].item(),
            "bbox": [bbox[0] * image_width, bbox[1] * image_height, 
                    (bbox[2] - bbox[0]) * image_width, (bbox[3] - bbox[1]) * image_height],
            "score": scores[i].item()
        })

print("原始预测框数量:", len(preds_coco_format))

# IOU计算函数
def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# 获取真实标注框
gt_boxes = []
for ann in anns:
    gt_boxes.append(ann['bbox'])

# 过滤IOU为零的预测框
filtered_preds_iou = []
for pred in preds_coco_format:
    pred_box = pred['bbox']
    max_iou = 0.0
    for gt_box in gt_boxes:
        iou = calculate_iou(pred_box, gt_box)
        max_iou = max(max_iou, iou)
    
    # 如果与任何真实框的IOU都大于0，则保留
    if max_iou > 0:
        filtered_preds_iou.append(pred)

print("过滤IOU为零后的预测框数量:", len(filtered_preds_iou))

# 过滤小面积标注函数
def filter_small_annotations(coco_dataset, min_area=144):
    """过滤掉面积小于min_area的标注"""
    filtered_annotations = []
    
    for ann in coco_dataset['annotations']:
        if ann['area'] >= min_area:
            filtered_annotations.append(ann)
    
    # 创建新的数据集
    filtered_dataset = {
        'images': coco_dataset['images'],
        'annotations': filtered_annotations,
        'categories': coco_dataset['categories']
    }
    
    return filtered_dataset

# 过滤小面积预测框函数
def filter_small_predictions(predictions, min_area=144):
    """过滤掉面积小于min_area的预测框"""
    filtered_predictions = []
    
    for pred in predictions:
        bbox = pred['bbox']
        area = bbox[2] * bbox[3]  # 宽 * 高
        if area >= min_area:
            filtered_predictions.append(pred)
    
    return filtered_predictions

# 第一次指标计算：原始预测结果与原始标注
print("\n=== 第一次指标计算：原始预测结果与原始标注 ===")
preds_path1 = 'preds_original.json'
with open(preds_path1, 'w') as f:
    json.dump(preds_coco_format, f)

coco_pred1 = coco.loadRes(preds_path1)
coco_eval1 = COCOeval(coco, coco_pred1, 'bbox')
coco_eval1.params.imgIds = [image_id]
coco_eval1.evaluate()
coco_eval1.accumulate()
coco_eval1.summarize()

# 第二次指标计算：筛选IOU为0后计算
print("\n=== 第二次指标计算：筛选IOU为0后计算 ===")
preds_path2 = 'preds_filtered_iou.json'
with open(preds_path2, 'w') as f:
    json.dump(filtered_preds_iou, f)

coco_pred2 = coco.loadRes(preds_path2)
coco_eval2 = COCOeval(coco, coco_pred2, 'bbox')
coco_eval2.params.imgIds = [image_id]
coco_eval2.evaluate()
coco_eval2.accumulate()
coco_eval2.summarize()

# 第三次指标计算：去除预测结果和原始标注中小于200平方像素的框
print("\n=== 第三次指标计算：去除预测结果和原始标注中小于200平方像素的框 ===")

# 创建过滤后的标注文件
filtered_annotations_path = 'filtered_annotations_area.json'
filtered_dataset = filter_small_annotations(coco.dataset, min_area=200)
with open(filtered_annotations_path, 'w') as f:
    json.dump(filtered_dataset, f)

# 过滤预测框中小于144平方像素的框
filtered_preds_area = filter_small_predictions(preds_coco_format, min_area=144)

print(f"原始标注数量: {len(coco.dataset['annotations'])}")
print(f"过滤小面积后的标注数量: {len(filtered_dataset['annotations'])}")
print(f"原始预测框数量: {len(preds_coco_format)}")
print(f"过滤小面积后的预测框数量: {len(filtered_preds_area)}")

# 使用过滤后的标注和预测结果进行评估
coco_filtered = COCO(filtered_annotations_path)
preds_path3 = 'preds_filtered_area.json'
with open(preds_path3, 'w') as f:
    json.dump(filtered_preds_area, f)

coco_pred3 = coco_filtered.loadRes(preds_path3)
coco_eval3 = COCOeval(coco_filtered, coco_pred3, 'bbox')
coco_eval3.params.imgIds = [image_id]
coco_eval3.evaluate()
coco_eval3.accumulate()
coco_eval3.summarize()

# 清理临时文件
os.remove(preds_path1)
os.remove(preds_path2)
os.remove(preds_path3)
os.remove(filtered_annotations_path)