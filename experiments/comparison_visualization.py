import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import datetime
from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util import box_ops
from PIL import Image
import datasets.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class SimpleVisualizer:
    def __init__(self):
        pass
    
    def visualize(self, img, pred_dict, save_path=None, dpi=150):
        """
        img: tensor(3, H, W) - 预处理后的图像
        pred_dict: 预测结果字典
        save_path: 保存路径，如果为None则不保存
        """
        # 创建图形，设置合适的尺寸
        fig = plt.figure(dpi=dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # 反归一化图像
        img = self.renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        
        # 添加预测框
        self.add_boxes(ax, pred_dict)
        
        if save_path is not None:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 保存图片，去除白边
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
            print(f"结果已保存至: {save_path}")
        
        plt.close(fig)
    
    def renorm(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """反归一化图像"""
        assert img.dim() == 3, "img.dim() should be 3 but %d" % img.dim()
        assert img.size(0) == 3, 'img.size(0) should be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        
        img_perm = img.permute(1, 2, 0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2, 0, 1)
    
    def add_boxes(self, ax, pred_dict):
        """添加边界框到图像"""
        if 'boxes' not in pred_dict or len(pred_dict['boxes']) == 0:
            return
        
        H, W = pred_dict['size'].tolist()
        boxes = pred_dict['boxes'].cpu()
        
        polygons = []
        for box in boxes:
            # 将归一化的cxcywh转换为像素坐标的xyxy
            cx, cy, w, h = box.tolist()
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            
            # 创建多边形
            poly = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
        
        # 创建红色框的集合
        if polygons:
            p = PatchCollection(polygons, 
                               facecolor='none', 
                               edgecolor=(1, 0, 0),  # 红色
                               linewidth=1,          # 线宽为1
                               linestyle='solid')
            ax.add_collection(p)

def main():
    # 路径配置
    model_config_path = "checkpoints/DINOpp_4scale.py"
    model_checkpoint_path = "checkpoints/checkpoint_best_regular.pth"
    annotation_path = "./datasets/validation/annotations/instances_filtered.json"
    image_path = "./datasets/validation/images/-63.39012861382869,-62.01824777399138,-19.75046718698471,-18.378586347147394.png"

    # 结果保存路径
    output_dir = "./detection_results"
    result_image_path = os.path.join(output_dir, "-63.39012861382869,-62.01824777399138,-19.75046718698471,-18.378586347147394.png")

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
    image_width, image_height = image.size
    print(f"原始图像尺寸: {image_width} x {image_height}")

    # 图像预处理
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_tensor, _ = transform(image, None)
    print("预处理后图像形状:", image_tensor.shape)

    # 图像预测
    with torch.no_grad():
        output = model.cuda()(image_tensor[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    # 使用新的可视化器
    threshold = 0.3
    vslzr = SimpleVisualizer()

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > threshold

    print(f"检测到 {len(scores)} 个目标，其中 {select_mask.sum()} 个超过阈值 {threshold}")

    box_label = [id2name[int(item)] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image_tensor.shape[1], image_tensor.shape[2]]),
        'box_label': box_label
    }

    # 保存结果图片（不显示，直接保存）
    print("正在保存检测结果图片...")
    vslzr.visualize(image_tensor, pred_dict, save_path=result_image_path, dpi=150)

    # 评估部分
    print("\n开始评估...")
    
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

    print(f"转换为COCO格式的预测结果数量: {len(preds_coco_format)}")

    # 将预测结果保存到临时JSON文件
    preds_path = 'preds_temp.json'
    with open(preds_path, 'w') as f:
        json.dump(preds_coco_format, f)

    # 加载预测结果并进行评估
    coco_pred = coco.loadRes(preds_path)
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = [image_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 保存评估结果到文本文件
    eval_result_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(eval_result_path, 'w', encoding='utf-8') as f:
        f.write("目标检测评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"图像文件: {os.path.basename(image_path)}\n")
        f.write(f"图像尺寸: {image_width} x {image_height}\n")
        f.write(f"检测阈值: {threshold}\n")
        f.write(f"总检测目标: {len(scores)}\n")
        f.write(f"超过阈值的目标: {select_mask.sum()}\n")
        f.write(f"COCO格式预测数量: {len(preds_coco_format)}\n")
        f.write("\nCOCO评估指标:\n")
        
        # 重定向COCOeval的输出到文件
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        coco_eval.summarize()
        
        sys.stdout = old_stdout
        eval_output = mystdout.getvalue()
        f.write(eval_output)
    
    print(f"评估结果已保存至: {eval_result_path}")

    # 清理临时文件
    os.remove(preds_path)
    print("临时文件已清理")

    # 打印总结信息
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"结果图片: {result_image_path}")
    print(f"评估结果: {eval_result_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()