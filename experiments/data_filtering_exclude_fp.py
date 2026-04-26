import sys
import io
import os
import shutil
import torch
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from main import build_model_main
from util.slconfig import SLConfig
import datasets.transforms as T
from tqdm import tqdm
import util.box_ops as box_ops

# 路径配置
model_config_path = "checkpoints/DINOpp_4scale.py"
model_checkpoint_path = "checkpoints/checkpoint_best_regular.pth"
annotation_paths = [
    "./datasets/transfer/annotations/train1.json",
    "./datasets/transfer/annotations/val1.json",
]
image_dirs = [
    "./datasets/transfer/images/",
    "./datasets/transfer/images/"
]
output_dir_high = "./数据集筛选结果/danet数据集/high/"
output_dir_low = "./数据集筛选结果/danet数据集/low/"
os.makedirs(output_dir_high + "images", exist_ok=True)
os.makedirs(output_dir_high + "annotations", exist_ok=True)
os.makedirs(output_dir_low + "images", exist_ok=True)
os.makedirs(output_dir_low + "annotations", exist_ok=True)

# 加载模型
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

# 过滤小面积标注函数
def filter_small_annotations(coco_dataset, min_area=200):
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

def evaluate_single_image_with_filters(image_path, coco, image_id, postprocessors):
    """使用IOU筛选和小面积过滤的评估函数"""
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    
    # 图像预处理和预测
    img_tensor, _ = transform(image, None)
    output = model.cuda()(img_tensor[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']
    threshold = 0.3
    
    # 获取原始预测结果
    preds_coco_format = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            bbox = boxes[i].cpu().numpy().tolist()
            preds_coco_format.append({
                "image_id": image_id,
                "category_id": labels[i].item(),
                "bbox": [bbox[0] * image_width, bbox[1] * image_height, 
                        (bbox[2] - bbox[0]) * image_width, (bbox[3] - bbox[1]) * image_height],
                "score": scores[i].item()
            })
    
    if len(preds_coco_format) == 0:
        print("该图片无预测结果:", os.path.basename(image_path), "--------------")
        return 0.0
    
    # 获取真实标注框
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    gt_boxes = [ann['bbox'] for ann in anns]
    
    # 过滤IOU为零的预测框
    filtered_preds_iou = []
    for pred in preds_coco_format:
        pred_box = pred['bbox']
        max_iou = 0.0
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
        
        if max_iou > 0:
            filtered_preds_iou.append(pred)
    
    # 过滤小面积预测框
    filtered_preds_area = filter_small_predictions(filtered_preds_iou, min_area=10)
    
    if len(filtered_preds_area) == 0:
        print("过滤后该图片无预测结果:", os.path.basename(image_path), "--------------")
        return 0.0
    
    # 创建过滤小面积后的标注
    filtered_dataset = filter_small_annotations(coco.dataset, min_area=10)
    
    # 创建临时文件进行评估
    preds_path = 'preds_filtered_tmp.json'
    anns_path = 'anns_filtered_tmp.json'
    
    with open(preds_path, 'w') as f:
        json.dump(filtered_preds_area, f)
    
    with open(anns_path, 'w') as f:
        json.dump(filtered_dataset, f)
    
    # 使用过滤后的标注和预测进行评估
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    
    coco_filtered = COCO(anns_path)
    coco_pred = coco_filtered.loadRes(preds_path)
    coco_eval = COCOeval(coco_filtered, coco_pred, 'bbox')
    coco_eval.params.imgIds = [image_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    sys.stdout = stdout_backup

    ap = coco_eval.stats[1]  # AP50
    
    # 清理临时文件
    os.remove(preds_path)
    os.remove(anns_path)
    
    return ap

def filter_and_save_all(annotation_paths, image_dirs, output_dir_high, output_dir_low):
    """处理所有数据集并合并结果"""
    all_high_anns, all_low_anns = [], []
    all_high_images, all_low_images = [], []
    all_categories = None
    
    # 新增区间统计字典
    ap_bins = {f"{i}-{i+5}%": 0 for i in range(0, 100, 5)}
    
    for annotation_path, image_dir in zip(annotation_paths, image_dirs):
        print(f"\n处理数据集: {annotation_path}")
        coco = COCO(annotation_path)
        with open(annotation_path, 'r') as f:
            ann_data = json.load(f)
            
        # 保存categories信息（所有数据集应该相同）
        if all_categories is None:
            all_categories = ann_data['categories']
            
        images = coco.dataset['images']
        
        for img in tqdm(images, desc=f"处理 {os.path.basename(annotation_path)}"):
            image_id = img['id']
            file_name = img['file_name']
            img_path = os.path.join(image_dir, file_name)
            if not os.path.exists(img_path):
                continue
                
            # 使用带过滤的评估函数
            ap = evaluate_single_image_with_filters(img_path, coco, image_id, postprocessors)
            
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)
            
            # 统计ap区间
            if 0.0 <= ap < 1.0:
                bin_idx = int((ap - 0.0) // 0.05)
                bin_key = f"{0 + bin_idx*5}-{0 + (bin_idx+1)*5}%"
                ap_bins[bin_key] += 1
                
            if ap > 0.7:
                shutil.copy(img_path, output_dir_high + "images/" + file_name)
                all_high_anns.extend(anns)
                all_high_images.append(img)
            else:
                shutil.copy(img_path, output_dir_low + "images/" + file_name)
                all_low_anns.extend(anns)
                all_low_images.append(img)
    
    # 保存合并后的标注
    def save_ann(anns, images, out_path):
        out_json = {
            "images": images,
            "annotations": anns,
            "categories": all_categories
        }
        with open(out_path, 'w') as f:
            json.dump(out_json, f)
    
    save_ann(all_high_anns, all_high_images, output_dir_high + "annotations/instances_filtered.json")
    save_ann(all_low_anns, all_low_images, output_dir_low + "annotations/instances_filtered.json")
    
    # 输出统计结果
    print(f"\n总体 AP区间统计:")
    for k, v in ap_bins.items():
        print(f"{k}: {v} 张图片")
    
    print(f"\n筛选结果:")
    print(f"高AP图片数量: {len(all_high_images)}")
    print(f"低AP图片数量: {len(all_low_images)}")

# 执行筛选
filter_and_save_all(annotation_paths, image_dirs, output_dir_high, output_dir_low)