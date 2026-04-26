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
model_config_path = "checkpoints/DINOpp_4scale.py"
model_checkpoint_path = "checkpoints/checkpoint_best_regular.pth"
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

# 同时过滤小面积和0 IOU的预测框函数
def filter_small_and_zero_iou_predictions(predictions, coco_gt, min_area=144):
    """同时过滤掉面积小于min_area且与任何真实框IOU为0的预测框"""
    filtered_predictions = []
    
    # 按图像分组处理
    image_ids = set([pred['image_id'] for pred in predictions])
    
    for img_id in tqdm(image_ids, desc="Filtering small area and zero IOU"):
        # 获取该图像的所有预测框
        img_preds = [pred for pred in predictions if pred['image_id'] == img_id]
        
        # 获取该图像的所有真实标注
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]
        
        # 过滤小面积和0 IOU的预测框
        for pred in img_preds:
            pred_box = pred['bbox']
            area = pred_box[2] * pred_box[3]
            
            # 如果面积小于阈值，跳过
            if area < min_area:
                continue
                
            # 计算与所有真实框的最大IOU
            max_iou = 0.0
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
            
            # 如果与任何真实框的IOU都大于0，则保留
            if max_iou > 0:
                filtered_predictions.append(pred)
    
    return filtered_predictions

# 第一次：原始预测结果
print("生成原始预测结果...")
threshold = 0.3
preds_coco_format = []

# 遍历所有图片
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

# 第二次：过滤IOU为0的预测框
print("过滤IOU为0的预测框...")
preds_filtered_iou = []

# 按图像分组处理
image_ids = set([pred['image_id'] for pred in preds_coco_format])

for img_id in tqdm(image_ids, desc="Filtering IOU=0"):
    # 获取该图像的所有预测框
    img_preds = [pred for pred in preds_coco_format if pred['image_id'] == img_id]
    
    # 获取该图像的所有真实标注
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_boxes = [ann['bbox'] for ann in anns]
    
    # 过滤IOU为0的预测框
    for pred in img_preds:
        pred_box = pred['bbox']
        max_iou = 0.0
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
        
        # 如果与任何真实框的IOU都大于0，则保留
        if max_iou > 0:
            preds_filtered_iou.append(pred)

# 第三次：过滤小面积的预测框
print("过滤小面积的预测框...")
preds_filtered_area = filter_small_predictions(preds_coco_format, min_area=10)

# 第四次：同时过滤小面积和0 IOU的预测框
print("同时过滤小面积和0 IOU的预测框...")
preds_filtered_both = filter_small_and_zero_iou_predictions(preds_coco_format, coco, min_area=10)

# 创建过滤小面积后的标注文件
print("创建过滤小面积后的标注文件...")
filtered_annotations_path = 'filtered_annotations_area.json'
filtered_dataset = filter_small_annotations(coco.dataset, min_area=10)
with open(filtered_annotations_path, 'w') as f:
    json.dump(filtered_dataset, f)

# 加载过滤后的COCO标注
coco_filtered = COCO(filtered_annotations_path)

print(f"原始预测框数量: {len(preds_coco_format)}")
print(f"过滤IOU为0后的预测框数量: {len(preds_filtered_iou)}")
print(f"过滤小面积后的预测框数量: {len(preds_filtered_area)}")
print(f"同时过滤小面积和0 IOU后的预测框数量: {len(preds_filtered_both)}")
print(f"原始标注数量: {len(coco.dataset['annotations'])}")
print(f"过滤小面积后的标注数量: {len(filtered_dataset['annotations'])}")

# 第一次指标计算：原始预测结果与原始标注
print("\n=== 第一次指标计算：原始预测结果与原始标注 ===")
preds_path1 = 'preds_original.json'
with open(preds_path1, 'w') as f:
    json.dump(preds_coco_format, f)

coco_pred1 = coco.loadRes(preds_path1)
coco_eval1 = COCOeval(coco, coco_pred1, 'bbox')
coco_eval1.evaluate()
coco_eval1.accumulate()
coco_eval1.summarize()

# 第二次指标计算：筛选IOU为0后计算
print("\n=== 第二次指标计算：筛选IOU为0后计算 ===")
preds_path2 = 'preds_filtered_iou.json'
with open(preds_path2, 'w') as f:
    json.dump(preds_filtered_iou, f)

coco_pred2 = coco.loadRes(preds_path2)
coco_eval2 = COCOeval(coco, coco_pred2, 'bbox')
coco_eval2.evaluate()
coco_eval2.accumulate()
coco_eval2.summarize()

# 第三次指标计算：去除预测结果和原始标注中小于144平方像素的框
print("\n=== 第三次指标计算：去除预测结果和原始标注中小于144平方像素的框 ===")
preds_path3 = 'preds_filtered_area.json'
with open(preds_path3, 'w') as f:
    json.dump(preds_filtered_area, f)

coco_pred3 = coco_filtered.loadRes(preds_path3)
coco_eval3 = COCOeval(coco_filtered, coco_pred3, 'bbox')
coco_eval3.evaluate()
coco_eval3.accumulate()
coco_eval3.summarize()

# 第四次指标计算：同时筛选小面积和0 IOU的预测框
print("\n=== 第四次指标计算：同时筛选小面积和0 IOU的预测框 ===")
preds_path4 = 'preds_filtered_both.json'
with open(preds_path4, 'w') as f:
    json.dump(preds_filtered_both, f)

coco_pred4 = coco_filtered.loadRes(preds_path4)
coco_eval4 = COCOeval(coco_filtered, coco_pred4, 'bbox')
coco_eval4.evaluate()
coco_eval4.accumulate()
coco_eval4.summarize()

# 清理临时文件
os.remove(preds_path1)
os.remove(preds_path2)
os.remove(preds_path3)
os.remove(preds_path4)
os.remove(filtered_annotations_path)