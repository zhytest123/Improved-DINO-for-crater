# 导入所需的库和模块
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test

# 定义一个函数来解析命令行参数
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)  # 配置文件路径
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')  # 可以覆盖配置文件中的设置

    # 数据集参数
    parser.add_argument('--dataset_file', default='coco')  # 数据集文件名
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')  # COCO数据集路径
    parser.add_argument('--coco_panoptic_path', type=str)  # COCO全景数据集路径
    parser.add_argument('--remove_difficult', action='store_true')  # 是否移除困难样本
    parser.add_argument('--fix_size', action='store_true')  # 是否固定图像大小

    # 训练参数
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')  # 输出目录
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')  # 实验备注
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')  # 使用的设备
    parser.add_argument('--seed', default=42, type=int)  # 随机种子
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 从检查点恢复训练
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')  # 预训练模型路径
    parser.add_argument('--finetune_ignore', type=str, nargs='+')  # 微调时忽略的参数
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')  # 起始训练轮次
    parser.add_argument('--eval', action='store_true')  # 是否进行评估
    parser.add_argument('--num_workers', default=10, type=int)  # 数据加载的工作线程数
    parser.add_argument('--test', action='store_true')  # 是否进行测试
    parser.add_argument('--debug', action='store_true')  # 是否开启调试模式
    parser.add_argument('--find_unused_params', action='store_true')  # 是否查找未使用的参数

    parser.add_argument('--save_results', action='store_true')  # 是否保存结果
    parser.add_argument('--save_log', action='store_true')  # 是否保存日志

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # 分布式进程数量
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')  # 分布式训练的URL
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')  # 分布式进程的排名
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')  # 本地进程排名
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")  # 是否使用混合精度训练
    
    return parser


def calculate_gflops(model, input_shape=(1, 3, 800, 1200)):
    """计算模型的GFLOPs"""
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # 临时设置模型为评估模式
        original_mode = model.training
        model.eval()
        
        with torch.no_grad():  # 确保不计算梯度
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        
        # 恢复原始模式
        model.train(original_mode)
        
        # 清理thop添加的hooks和属性
        def clean_hooks(module):
            if hasattr(module, '_buffers'):
                keys_to_remove = []
                for key in module._buffers.keys():
                    if 'total_ops' in key or 'total_params' in key:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del module._buffers[key]
            
            # 清理属性
            if hasattr(module, 'total_ops'):
                delattr(module, 'total_ops')
            if hasattr(module, 'total_params'):
                delattr(module, 'total_params')
            
            # 清理所有hooks
            module._forward_hooks.clear()
            module._backward_hooks.clear()
            
            # 递归清理子模块
            for child in module.children():
                clean_hooks(child)
        
        # 清理整个模型
        clean_hooks(model)
        
        gflops = flops / 1e9
        return gflops
    except ImportError:
        print("Warning: thop not installed, cannot calculate GFLOPs")
        return 0.0
    except Exception as e:
        print(f"Warning: Error calculating GFLOPs: {e}")
        # 确保即使出错也要清理hooks
        try:
            def emergency_clean(module):
                module._forward_hooks.clear()
                module._backward_hooks.clear()
                for child in module.children():
                    emergency_clean(child)
            emergency_clean(model)
        except:
            pass
        return 0.0


def log_metrics_to_dino_log(output_dir, epoch, test_stats, n_parameters, gflops=0.0):
    """记录详细的评估指标到dino_log.txt"""
    dino_log_path = Path(output_dir) / "dino_log.txt"
    
    # 提取COCO评估指标
    metrics_info = {
        'epoch': epoch,
        'timestamp': str(datetime.datetime.now()),
        'params': n_parameters,
        'gflops': gflops,
    }
    
    # 如果有COCO bbox评估结果
    if 'coco_eval_bbox' in test_stats:
        coco_stats = test_stats['coco_eval_bbox']
        metrics_info.update({
            'AP_all': coco_stats[0],           # AP @ IoU=0.50:0.95
            'AP_50': coco_stats[1],            # AP @ IoU=0.50
            'AP_75': coco_stats[2],            # AP @ IoU=0.75
            'AP_small': coco_stats[3],         # AP for small objects
            'AP_medium': coco_stats[4],        # AP for medium objects
            'AP_large': coco_stats[5],         # AP for large objects
            'AR_1': coco_stats[6],             # AR @ maxDets=1
            'AR_10': coco_stats[7],            # AR @ maxDets=10
            'AR_100': coco_stats[8],           # AR @ maxDets=100
            'AR_small': coco_stats[9],         # AR for small objects
            'AR_medium': coco_stats[10],       # AR for medium objects
            'AR_large': coco_stats[11],        # AR for large objects
        })

    # 如果有分割评估结果
    if 'coco_eval_masks' in test_stats:
        mask_stats = test_stats['coco_eval_masks']
        metrics_info.update({
            'AP_mask_all': mask_stats[0],
            'AP_mask_50': mask_stats[1],
            'AP_mask_75': mask_stats[2],
            'AR_mask_100': mask_stats[8],
        })
    
    # 添加其他训练指标
    for key, value in test_stats.items():
        if key not in ['coco_eval_bbox', 'coco_eval_masks'] and isinstance(value, (int, float)):
            metrics_info[f'test_{key}'] = value
    
    # 写入文件
    with open(dino_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics_info, indent=2) + "\n")
        f.write("-" * 80 + "\n")


# 构建模型的主函数
def build_model_main(args):
    # 使用注册机制来维护模型
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

# 主函数
def main(args):
    utils.init_distributed_mode(args)  # 初始化分布式模式
    # 加载配置文件并更新参数
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # 更新一些临时参数
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # 设置日志记录器
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)  # 设置设备

    # 固定随机种子以保证可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # 指数移动平均
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

        # 计算GFLOPs (只在主进程中计算一次)
    gflops = 0.0
    if utils.is_main_process():
        gflops = calculate_gflops(model_without_ddp)
        logger.info(f'Model GFLOPs: {gflops:.2f}')


    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # 构建训练和验证数据集
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # 学习率调度器
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 如果是COCO全景数据集，进行额外的AP评估
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    # 如果有冻结的权重，加载它们
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # 如果没有恢复训练并且有预训练模型路径，加载预训练模型
    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        

    # 如果设置了评估标志，进行评估
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        # 记录评估结果到dino_log.txt
        if args.output_dir and utils.is_main_process():
            log_metrics_to_dino_log(args.output_dir, 0, test_stats, n_parameters, gflops)

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # 开始训练
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)

    # 初始化dino_log.txt文件（只在主进程中）
    if args.output_dir and utils.is_main_process():
        dino_log_path = Path(args.output_dir) / "dino_log.txt"
        with open(dino_log_path, 'w', encoding='utf-8') as f:
            f.write(f"DINO Training Log - Started at {datetime.datetime.now()}\n")
            f.write(f"Model Parameters: {n_parameters:,}\n")
            f.write(f"Model GFLOPs: {gflops:.2f}\n")
            f.write("=" * 80 + "\n")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # 在学习率下降前和每100轮次保存检查点
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # 评估模型
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )

         # 记录详细指标到dino_log.txt（只在主进程中）
        if args.output_dir and utils.is_main_process():
            log_metrics_to_dino_log(args.output_dir, epoch, test_stats, n_parameters, gflops)
        
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # 评估EMA模型
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )

            # 记录EMA模型的详细指标到dino_log.txt
            if args.output_dir and utils.is_main_process():
                # 为EMA结果添加前缀
                ema_test_stats_prefixed = {f'ema_{k}': v for k, v in ema_test_stats.items()}
                log_metrics_to_dino_log(args.output_dir, epoch, ema_test_stats_prefixed, n_parameters, gflops)
            
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 保存评估日志
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 移除复制的文件
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

# 主程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
