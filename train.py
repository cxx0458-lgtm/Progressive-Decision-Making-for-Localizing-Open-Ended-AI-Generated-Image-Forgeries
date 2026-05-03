import os
import json
import time
import types
import inspect
import argparse
import torch
import datetime
import numpy as np
from argparse import Namespace
from pathlib import Path
from torch.serialization import add_safe_globals
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.transforms import get_albu_transforms

from IMDLBenCo.datasets import ManiDataset, JsonDataset, BalancedDataset
from IMDLBenCo.evaluation import PixelF1, ImageF1 # TODO You can select evaluator you like here

from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.training_scripts.trainer import train_one_epoch

from progressive_mesorch import ProgressiveMesorch

def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo training launch!', add_help=True)
    # ++++++++++++TODO++++++++++++++++
    # 这里是每个模型定制化的input区域，包括load与训练模型，模型的magic number等等
    # 需要根据你们的模型定制化修改这里 
    # 目前这里的内容都是仅仅给IML-ViT用的
    # parser.add_argument('--vit_pretrain_path', default = '/root/workspace/IML-ViT/pretrained-weights/mae_pretrain_vit_base.pth', type=str, help='path to vit pretrain model by MAE')

    # parser.add_argument('--edge_lambda', default=20, type=float,
    #                     help='hyper-parameter of the weight for proposed edge loss.')
    # parser.add_argument('--predict_head_norm', default="BN", type=str,
    #                     help="norm for predict head, can be one of 'BN', 'LN' and 'IN' (batch norm, layer norm and instance norm). It may influnce the result  on different machine or datasets!")
    # -------------------------------
    # Model name
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    # ++++++++++++ 新增：Look Twice (Soft-Seed Growing) 参数 ++++++++++++
    parser.add_argument('--use_look_twice', action='store_true',
                        help='Enable Look Twice (Soft-Seed Growing) module in the model.')
    parser.add_argument('--dice_weight', default=0.5, type=float,
                        help='Dice loss weight for final fused prediction.')
    parser.add_argument('--lt_deep_dice_weight', default=0.3, type=float,
                        help='Dice loss weight for deep supervision in Look-Twice.')
    parser.add_argument('--lt_steps', default=3, type=int,
                        help='Iterations for Soft-Seed growing.')
    parser.add_argument('--lt_tau_low', default=0.2, type=float,
                        help='Lower bound threshold for ambiguous candidate regions.')
    parser.add_argument('--lt_tau_high', default=0.8, type=float,
                        help='Upper bound threshold for confident seed regions.')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze main model backbone, ONLY fine-tune the LookTwice module.')
    parser.add_argument('--finetune', default='', type=str,
                        help='Path to checkpoint for fine-tuning. It will load weights with strict=False.')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 可以接受label的模型是否接受label输入，并启用相关的loss。
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str,
                        help='dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    # ------------------------------------
    # training related
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--epochs', default=200, type=int)
    # Test related
    parser.add_argument('--no_model_eval', action='store_true', 
                        help='Do not use model.eval() during testing.')
    parser.add_argument('--test_period', default=4, type=int,
                        help="how many epoch per testing one time")
    
    
    
    # 一个epoch在tensorboard中打几个loss的data point
    parser.add_argument('--log_per_epoch_count', default=20, type=int,
                        help="how many loggings (data points for loss) per testing epoch in Tensorboard")
    
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='find_unused_parameters for DDP. Mainly solve issue for model with image-level prediction but not activate during training.')
    
    # 不启用AMP（自动精度）进行训练
    parser.add_argument('--if_not_amp', action='store_false',
                        help='Do not use automatic precision.')
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # ----输出的日志相关的参数-----------
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint, input the path of a ckpt.')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args, remaining_args = parser.parse_known_args()
     # 获取对应的模型类
    model_class = MODELS.get(args.model)

    # 根据模型类动态创建参数解析器
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[WARN] Failed to remove {path}: {e}")


def save_checkpoint_to_path(args, model, model_without_ddp, optimizer, loss_scaler, epoch, ckpt_path):
    """
    先让原始 save_model 正常保存 checkpoint-{epoch}.pth，
    然后重命名/移动到你想要的路径。
    """
    misc.save_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        epoch=epoch
    )

    default_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
    if default_path != ckpt_path and os.path.exists(default_path):
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        os.replace(default_path, ckpt_path)

def main(args, model_args):
    # init parameters for distributed training
    add_safe_globals([Namespace])
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = get_albu_transforms('train')
    test_transform = get_albu_transforms('test')

    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    # ---- dataset with crop augmentation ----
    if os.path.isdir(args.data_path):
        dataset_train = ManiDataset(
            args.data_path, 
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=train_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
    else:
        try:
            dataset_train = JsonDataset(
                args.data_path, 
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=train_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        except:
            dataset_train = BalancedDataset(
                args.data_path,    
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=train_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
    
    if os.path.isdir(args.test_data_path):
        dataset_test = ManiDataset(
            args.test_data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )

    else:
        dataset_test = JsonDataset(
            args.test_data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
    # ------------------------------------
    
    print(dataset_train)
    print(dataset_test)
    global_rank = 0
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # ========define the model directly==========
    # model = IML_ViT(
    #     vit_pretrain_path = model_args.vit_pretrain_path,
    #     predict_head_norm= model_args.predict_head_norm,
    #     edge_lambda = model_args.edge_lambda
    # )
    
    # --------------- or -------------------------
    # Init model with registry
    model = MODELS.get(args.model)
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    for k, v in vars(model_args).items():
        if k in model_init_params and k not in combined_args:
            combined_args[k] = v
    model = model(**combined_args)
    # ============================================
    # +++++++++++++++ 新增：加载预训练权重用于微调 +++++++++++++++
    if args.finetune:
        print(f"\n====== 正在从 {args.finetune} 加载预训练权重 (strict=False) ======")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        # 允许 key 不匹配 (这样新增的 look_twice 就会被忽略并随机初始化)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"缺失的参数 (对于新增的 LookTwice 模块这是正常的): {missing_keys}\n")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # evaluator_list = [
    #     PixelF1(threshold=0.5, mode="origin"),
    #     # ImageF1(threshold=0.5)
    # ]
    # evaluator_list = [
    #     PixelF1(threshold=0.5, mode="origin"),
    #     PixelF1(threshold=0.5, mode="double"),
    #     # ImageF1(threshold=0.5)
    # ]
    origin_f1 = PixelF1(threshold=0.5, mode="origin")
    double_f1 = PixelF1(threshold=0.5, mode="double")

    origin_f1.name = "pixel-level F1"
    double_f1.name = "Permute F1-score"

    evaluator_list = [
        origin_f1,
        double_f1,
    ]
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    args.opt='AdamW'
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    # ++++++++++++ 新增：冻结骨干网络，只训练 LookTwice 模块 ++++++++++++
    if args.freeze_backbone and args.use_look_twice:
        print(">>> 冻结骨干网络！只训练 'look_twice' 模块...")
        for name, param in model_without_ddp.named_parameters():
            if 'look_twice' in name or 'sparse_attn' in name or 'fusion_weight' in name:
                param.requires_grad = True
                print(f"    [参与训练]: {name}")
            else:
                param.requires_grad = False
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    optimizer  = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_evaluate_metric_value = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            log_per_epoch_count=args.log_per_epoch_count,
            args=args
        )

        optimizer.zero_grad()

        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            # test_stats = test_one_epoch(
            #     model,
            #     data_loader=data_loader_test,
            #     evaluator_list=evaluator_list,
            #     device=device,
            #     epoch=epoch,
            #     log_writer=log_writer,
            #     args=args
            # )
            # ===== test 阶段把 tau_low / tau_high 都减 0.2 =====
            model_for_tau = model.module if hasattr(model, "module") else model

            old_tau_low = float(model_for_tau.look_twice.tau_low)
            old_tau_high = float(model_for_tau.look_twice.tau_high)

            model_for_tau.look_twice.tau_low = max(0.0, old_tau_low - 0.2)
            model_for_tau.look_twice.tau_high = max(0.0, old_tau_high - 0.2)

            if misc.is_main_process():
                print(
                    f"[TEST TAU] train=({old_tau_low:.3f}, {old_tau_high:.3f}) "
                    f"test=({model_for_tau.look_twice.tau_low:.3f}, {model_for_tau.look_twice.tau_high:.3f})"
                )

            test_stats = test_one_epoch(
                model,
                data_loader=data_loader_test,
                evaluator_list=evaluator_list,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args
            )

            # ===== test 完恢复训练用的 tau，避免影响下一轮训练 =====
            model_for_tau.look_twice.tau_low = old_tau_low
            model_for_tau.look_twice.tau_high = old_tau_high

            evaluate_metric_for_ckpt = evaluator_list[0].name   # pixel-level F1
            evaluate_metric_value = test_stats[evaluate_metric_for_ckpt]

            if args.output_dir and misc.is_main_process():
                # 先把当前epoch存成临时文件
                current_tmp_ckpt = os.path.join(args.output_dir, f"tmp_epoch_{epoch}.pth")
                save_checkpoint_to_path(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    ckpt_path=current_tmp_ckpt
                )

                # 如果当前更好，就替换 best_model.pth
                if evaluate_metric_value > best_evaluate_metric_value:
                    print(f"Best {evaluate_metric_for_ckpt}: {best_evaluate_metric_value:.6f} -> {evaluate_metric_value:.6f} at epoch {epoch}")
                    best_evaluate_metric_value = evaluate_metric_value
                    best_epoch = epoch

                    if os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)
                    os.replace(current_tmp_ckpt, best_ckpt_path)
                else:
                    # 当前不是best，直接删掉
                    safe_remove(current_tmp_ckpt)

            else:
                # 非主进程也更新一下变量，保证日志一致
                if evaluate_metric_value > best_evaluate_metric_value:
                    best_evaluate_metric_value = evaluate_metric_value
                    best_epoch = epoch

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_pixel_f1': best_evaluate_metric_value,
            }
        else:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_pixel_f1': best_evaluate_metric_value,
            }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    print(f"Final best pixel-level F1 = {best_evaluate_metric_value:.6f} at epoch {best_epoch}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)