import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util import box_ops
from PIL import Image
import datasets.transforms as T
import glob

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
                               linewidth=2,          # 线宽为1
                               linestyle='solid')
            ax.add_collection(p)

def process_single_image(model, postprocessors, transform, vslzr, id2name, 
                         image_path, output_dir, threshold=0.3):
    """处理单张图片并进行预测"""
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        
        # 图像预处理
        image_tensor, _ = transform(image, None)
        
        # 图像预测
        with torch.no_grad():
            output = model.cuda()(image_tensor[None].cuda())
            output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        
        # 过滤低置信度预测
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > threshold
        
        # 获取类别标签
        box_label = [id2name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([image_tensor.shape[1], image_tensor.shape[2]]),
            'box_label': box_label
        }
        
        # 生成输出路径
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        result_image_path = os.path.join(output_dir, f"{name_without_ext}_detected.png")
        
        # 保存结果图片
        vslzr.visualize(image_tensor, pred_dict, save_path=result_image_path, dpi=150)
        
        # 打印检测信息
        detected_count = select_mask.sum().item()
        print(f"图片: {filename} - 检测到 {detected_count} 个目标")
        
        return {
            'filename': filename,
            'detected_count': detected_count,
            'total_predictions': len(scores),
            'scores': scores[select_mask].cpu().numpy().tolist(),
            'labels': box_label
        }
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None

def main():
    # 路径配置
    model_config_path = "ckpts_dppp/row.py"
    model_checkpoint_path = "ckpts_dppp/checkpoint_best_regular.pth"
    input_image_dir = "C:\\Users\\zhy\\Desktop\\电塔数据集\\EPD_YOLO\\images\\val"  # 输入图片文件夹
    output_dir = "./detection_results"  # 输出结果文件夹
    
    # 检测阈值
    threshold = 0.3
    
    # 支持的图片格式
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    print("开始初始化模型...")
    
    # 加载模型配置和检查点
    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    # 加载类别名称
    with open('util/crater_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k): v for k, v in id2name.items()}

    # 图像预处理
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 初始化可视化器
    vslzr = SimpleVisualizer()
    
    # 获取所有图片文件
    image_files = []
    for format in supported_formats:
        image_files.extend(glob.glob(os.path.join(input_image_dir, format)))
        image_files.extend(glob.glob(os.path.join(input_image_dir, format.upper())))
    
    if not image_files:
        print(f"在文件夹 {input_image_dir} 中没有找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 处理结果统计
    results_summary = {
        'total_images': len(image_files),
        'processed_images': 0,
        'failed_images': 0,
        'total_detections': 0,
        'threshold': threshold,
        'processing_time': None
    }
    
    detailed_results = []
    
    import time
    start_time = time.time()
    
    # 批量处理图片
    for i, image_path in enumerate(image_files):
        print(f"\n处理进度: {i+1}/{len(image_files)}")
        result = process_single_image(model, postprocessors, transform, vslzr, 
                                    id2name, image_path, output_dir, threshold)
        
        if result is not None:
            results_summary['processed_images'] += 1
            results_summary['total_detections'] += result['detected_count']
            detailed_results.append(result)
        else:
            results_summary['failed_images'] += 1
    
    # 计算处理时间
    end_time = time.time()
    results_summary['processing_time'] = end_time - start_time
    
    # 保存处理摘要
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': results_summary,
            'detailed_results': detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    # 打印总结信息
    print("\n" + "=" * 60)
    print("批量处理完成!")
    print(f"处理图片总数: {results_summary['total_images']}")
    print(f"成功处理: {results_summary['processed_images']}")
    print(f"处理失败: {results_summary['failed_images']}")
    print(f"总检测目标数: {results_summary['total_detections']}")
    print(f"平均每张图片检测数: {results_summary['total_detections']/max(1, results_summary['processed_images']):.2f}")
    print(f"总处理时间: {results_summary['processing_time']:.2f} 秒")
    print(f"平均每张图片处理时间: {results_summary['processing_time']/max(1, results_summary['processed_images']):.2f} 秒")
    print(f"结果保存路径: {output_dir}")
    print(f"处理摘要: {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()