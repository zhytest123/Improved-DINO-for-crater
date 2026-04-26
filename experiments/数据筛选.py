import sys
import io
import os
import shutil
import torch
import json
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from main import build_model_main
from util.slconfig import SLConfig
import datasets.transforms as T
from tqdm import tqdm


# 路径配置
model_config_path = "ckpts/dinopp4/DINOpp_4scale.py"
model_checkpoint_path = "ckpts/dinopp4/checkpoint_best_regular.pth"
annotation_paths = [
    "./lronac数据集/annotations/instances_filtered.json",
]
image_dirs = [
    "./lronac数据集/images/",
]
output_dir_high = "./数据集筛选结果/nac数据集/high_ap/"
output_dir_low = "./数据集筛选结果/nac数据集/low_ap/"
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

def evaluate_single_image(image_path, coco, image_id, postprocessors):
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    img_tensor, _ = transform(image, None)
    output = model.cuda()(img_tensor[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']
    threshold = 0.3
    preds_coco_format = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            bbox = boxes[i].cpu().numpy().tolist()
            preds_coco_format.append({
                "image_id": image_id,
                "category_id": labels[i].item(),
                "bbox": [bbox[0] * image_width, bbox[1] * image_height, (bbox[2] - bbox[0]) * image_width, (bbox[3] - bbox[1]) * image_height],
                "score": scores[i].item()
            })
    if len(preds_coco_format) == 0:
        print("该图片无预测结果:", os.path.basename(image_path), "--------------")
        return 0.0  # 没有检测结果直接返回0
    preds_path = 'preds_tmp.json'
    with open(preds_path, 'w') as f:
        json.dump(preds_coco_format, f)
    
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    coco_pred = coco.loadRes(preds_path)
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = [image_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    sys.stdout = stdout_backup

    #ap = coco_eval.stats[0] # AP50:95
    ap = coco_eval.stats[1] # AP50
    os.remove(preds_path)
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
                
            ap = evaluate_single_image(img_path, coco, image_id, postprocessors)
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)
         # 统计ap区间
            if 0.0 <= ap < 1.0:
                bin_idx = int((ap - 0.0) // 0.05)
                bin_key = f"{0 + bin_idx*5}-{0 + (bin_idx+1)*5}%"
                ap_bins[bin_key] += 1
                
            if ap > 0.99:
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


# 替换原来的循环调用
filter_and_save_all(annotation_paths, image_dirs, output_dir_high, output_dir_low)