from functools import partial

import time
import pickle
import argparse

import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.backends import cuda

import wandb
import cv2

from datasets import load_dataset

from src.transforms import *
from src.utils import *
from src.loss import *
from src.net import Net
from src.run import train, evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CFIRSTNET')
    parser.add_argument('--seed', type=int, default=1224, help='random seed')

    parser.add_argument('--wandb', type=bool, default=True, help='use wandb')
    parser.add_argument('--project', type=str, default='CFIRSTNET', help='project name')
    parser.add_argument('--name', type=str, default='convnextv2_tiny_CF_BeGAN', help='experiemnt name')

    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    #parser.add_argument('--use_BeGAN', type=bool, default=True, help='use BeGAN dataset')

    parser.add_argument('--img_size', type=int, default=384, help='image size')
    parser.add_argument('--interpolation', type=str, default='area', help='interpolation method, options:[area, linear, cubic, nearest]')

    parser.add_argument('--use_PDN', type=bool, default=False, help='use PDN density map')
    parser.add_argument('--use_dist', type=bool, default=False, help='use effective distance map')
    parser.add_argument('--use_current', type=bool, default=False, help='use current map')
    parser.add_argument('--use_HIRD', type=bool, default=True, help='use Hypothetical IR Drop map')
    parser.add_argument('--use_WR', type=bool, default=True, help='use Wire Resistance map')
    parser.add_argument('--use_RD', type=bool, default=True, help='use Resistive Distance map')

    parser.add_argument('--metal', type=str, default="m1,m4,m7,m8,m9", help='metal layers in netlist')
    parser.add_argument('--metal_orient', type=str, default="V,H,V,H,V", help='metal orientation in netlist, V for vertical, H for horizontal')

    parser.add_argument('--backbone', type=str, default='convnextv2_tiny.fcmae', help='model backbone from timm')
    parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained backbone from timm')
    parser.add_argument('--stochastic_depth', type=float, default=0.5, help='stochastic depth')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--decoder_channels', type=int, default=128, help='decoder channels')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=None, help='clip grad norm')

    parser.add_argument('--cycle_limit', type=int, default=1, help='cycle limit')

    parser.add_argument('--train_batch_size', type=int, default=10, help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=100, help='gradient accumulation steps')

    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--use_amp', type=bool, default=True, help='use automatic mixed precision')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')

    parser.add_argument('--save_model', type=bool, default=True, help='save best MAE model in validation set')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save directory')
    parser.add_argument('--save_best_model', action='store_true', help='Save the best model based on validation performance')

    args = parser.parse_args()

    metal = args.metal.split(',')
    metal_num = len(metal)

    metal_orient = args.metal_orient.split(',')
    metal_orient = [2 if o == 'V' else 3 if o == 'H' else 0 for o in metal_orient]
    if len(metal_orient) != metal_num or metal_orient.count(0) > 0:
        raise ValueError('Invalid metal definition')

    #in_channels = args.use_PDN + args.use_dist + args.use_current + args.use_HIRD * (metal_num * 2 - 1) + args.use_WR * (metal_num * 2 - 3) + args.use_RD * (metal_num * 2 - 3)
    

    if args.interpolation == 'area':
        interpolation = cv2.INTER_AREA
    elif args.interpolation == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif args.interpolation == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif args.interpolation == 'nearest':
        interpolation = cv2.INTER_NEAREST
    else:
        raise ValueError('Invalid interpolation method')
    
    # if args.use_gpu:
    #     device = torch.device(f'cuda:{args.gpu}')
    # else:
    #     device = torch.device('cpu')

    # if args.use_gpu:
    #     torch.cuda.set_device(args.gpu)

    #     cudnn.enabled = True
    #     cudnn.benchmark = True
    #     cuda.matmul.allow_tf32 = True
    #     cudnn.allow_tf32 = True
        
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    
        cudnn.enabled = True
        cudnn.benchmark = True
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
    else:
        print("GPU not available. Falling back to CPU.")
        device = torch.device('cpu')
        args.use_gpu = False  # 이후 AMP 등에서 참조할 경우 대비

    if args.wandb:
        wandb.init(project=args.project, name=args.name, save_code=True)

    # set seed
    seed_everything(args.seed)

    # create dataset
    train_dataset = load_dataset(
        path = 'dataset.py',
        test_mode = False,
        #use_BeGAN = args.use_BeGAN,
        
        img_size = args.img_size,
        interpolation = interpolation,
        
        split = 'fake',
        num_proc = args.num_workers,
        keep_in_memory = False,
        writer_batch_size = 10,
        trust_remote_code=True,
    )
    train_dataset = train_dataset.with_format('numpy')
    train_dataset = train_dataset.map(data_mapping, batched=True, batch_size=1, num_proc=args.num_workers, remove_columns=train_dataset.column_names)
    in_channels = train_dataset[0]['image'].shape[0]
    args.in_channels = in_channels

    valid_dataset = load_dataset(
        path = 'dataset.py',
        test_mode = False,
        #use_BeGAN = args.use_BeGAN,
        
        img_size = args.img_size,
        interpolation = interpolation,
        
        split = 'real',
        num_proc = args.num_workers,
        keep_in_memory = False,
        writer_batch_size = 10,
        trust_remote_code=True,
    )
    valid_dataset = valid_dataset.with_format('numpy')
    valid_dataset = valid_dataset.map(data_mapping, batched=True, batch_size=1, num_proc=args.num_workers, remove_columns=valid_dataset.column_names)


    test_dataset = load_dataset(
        path = 'dataset.py',
        test_mode = False,
        #use_BeGAN = args.use_BeGAN,
        
        img_size = args.img_size,
        interpolation = interpolation,
    
        split = 'test',
        num_proc = args.num_workers,
        keep_in_memory = False,
        writer_batch_size = 10,
        trust_remote_code=True,
    )
    test_dataset = test_dataset.with_format('numpy')
    test_dataset = test_dataset.map(data_mapping, batched=True, batch_size=1, num_proc=args.num_workers, remove_columns=test_dataset.column_names)

    try:
        with open('min_max_mean_std.cache', 'rb') as f:
            min, max, mean, std = pickle.load(f)
        print('Load cache Successfully')
    except:
        #min, max, mean, std = get_min_max_mean_std(train_dataset, in_chs=[in_channels, 1], data_keys=['image', 'ir_drop'])
        min, max, mean, std = get_min_max_mean_std(train_dataset, in_chs=[train_dataset[0]['image'].shape[0], 1], data_keys=['image', 'ir_drop'])
        with open('min_max_mean_std.cache', 'wb') as f:
            pickle.dump((min, max, mean, std), f)

    train_ds = train_dataset.with_format('numpy')
    valid_ds = valid_dataset.with_format('numpy')
    test_ds  = test_dataset.with_format('numpy')
    train_ds = train_ds.map(partial(normalize, mean=mean, std=std), batched=True, batch_size=1, num_proc=args.num_workers, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(partial(normalize, mean=mean, std=std), batched=True, batch_size=1, num_proc=args.num_workers, remove_columns=valid_ds.column_names)
    test_ds  = test_ds.map (partial(normalize, mean=mean, std=std), batched=True, batch_size=1, num_proc=args.num_workers, remove_columns=test_ds.column_names)
    train_ds = train_ds.with_format('torch')
    valid_ds = valid_ds.with_format('torch')
    test_ds  = test_ds.with_format('torch')

    train_loader = DataLoader(
        dataset=train_ds,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_ds,
        collate_fn=collate_fn,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        collate_fn=collate_fn,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = Net(
        model_backbone=args.backbone,
        model_pretrained=args.pretrained,
        in_channels=args.in_channels,
        stochastic_depth=args.stochastic_depth,
        dropout=args.dropout,
        decoder_channels=args.decoder_channels,
        out_channels=1,
    ).to(device)
    
    # start training
    start = time.time()
    model = train(args, model, train_loader, valid_loader, test_loader, mean, std, device)
        
    # end training
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    print('---------------------------------------------------------------------------------------')
    
    # 
    print('Evaluation start')
    evaluate(args, model, valid_loader, test_loader, mean, std, device)
