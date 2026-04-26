import os
import torch
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from main import build_model_main
from util.slconfig import SLConfig
from util import box_ops
import datasets.transforms as T
from pycocotools.coco import COCO

# 路径配置
model_config_path = "ckpts/dinopp4/DINOpp_4scale.py"
model_checkpoint_path = "ckpts/dinopp4/checkpoint_best_regular.pth"
annotation_path = "./验证数据集/annotations/instances_filtered.json"
image_path = "./验证数据集/images/172.57337583818827,173.94525667802563,-18.378586347147394,-17.006705507310105.png"
output_path = "改进DINO_172.57337583818827,173.94525667802563,-18.378586347147394,-17.006705507310105.png"

def visualize_detection(image, pred_boxes, true_boxes, output_path=None):
    """
    可视化检测结果
    image: PIL Image
    pred_boxes: list of [x1, y1, x2, y2] 预测框
    true_boxes: list of [x, y, w, h] 真实框 (COCO格式)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 显示图像
    ax.imshow(image)
    
    # 绘制预测框 (绿色实线)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=1, edgecolor=(0,1,0), facecolor='none', linestyle='-')
        ax.add_patch(rect)
    
    # 绘制真实框 (红色虚线)
    for box in true_boxes:
        x, y, w, h = box
        rect = Rectangle((x, y), w, h, 
                        linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
    
    # 移除坐标轴
    ax.set_axis_off()
    
    # 移除边框和空白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150, facecolor='white')
        print(f"可视化结果已保存到: {output_path}")
    
    plt.close()

def main():
    print("=== DINO 检测结果可视化 ===")
    
    # 检查文件是否存在
    for path in [model_config_path, model_checkpoint_path, annotation_path, image_path]:
        if not os.path.exists(path):
            print(f"错误: 路径不存在: {path}")
            return
    
    # 加载模型配置和检查点
    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    # 加载COCO类别名称
    with open('util/crater_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k): v for k, v in id2name.items()}
    
    # 加载图像
    original_image = Image.open(image_path).convert("RGB")
    image_width, image_height = original_image.size
    
    # 图像预处理
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor, _ = transform(original_image, None)
    print(f"预处理后图像形状: {image_tensor.shape}")
    
    # 图像预测
    with torch.no_grad():
        output = model(image_tensor[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    
    # 设置阈值并筛选预测结果
    threshold = 0.3
    scores = output['scores']
    boxes = output['boxes']
    
    # 获取预测框 (xyxy格式，像素坐标)
    pred_boxes = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            bbox = boxes[i].cpu().numpy()
            # 转换为像素坐标
            x1 = bbox[0] * image_width
            y1 = bbox[1] * image_height
            x2 = bbox[2] * image_width
            y2 = bbox[3] * image_height
            pred_boxes.append([x1, y1, x2, y2])
    
    print(f"检测到 {len(pred_boxes)} 个预测框")
    
    # 加载COCO注释并获取真实标注框
    coco = COCO(annotation_path)
    
    # 从COCO注释中获取图像ID
    image_id = None
    for img in coco.dataset['images']:
        if img['file_name'] == os.path.basename(image_path):
            image_id = img['id']
            break
    
    if image_id is None:
        print("警告: 在COCO注释中未找到图像，只显示预测框")
        true_boxes = []
    else:
        # 获取图像的真实标注
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        
        # 获取真实标注框 (xywh格式，像素坐标)
        true_boxes = [ann['bbox'] for ann in anns]
        print(f"找到 {len(true_boxes)} 个真实标注框")
    
    # 可视化结果
    visualize_detection(original_image, pred_boxes, true_boxes, output_path)
    
    print(f"\n=== 处理完成 ===")
    print(f"预测框: {len(pred_boxes)} 个 (绿色实线)")
    print(f"真实框: {len(true_boxes)} 个 (红色虚线)")

if __name__ == "__main__":
    main()