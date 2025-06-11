import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

from timm.scheduler.cosine_lr import CosineLRScheduler

import wandb

from src.transforms import *
from src.utils import *
from src.loss import *
from src.metrics import get_exceeding_indices, map_to_physical_coordinates

def train(args, model, train_loader, valid_loader, test_loader, mean, std, device):
    # loss function
    criterion = RMSELoss()
    aux_criterion = DiceLoss(mean=mean['ir_drop'], std=std['ir_drop'])

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # scheduler
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=args.epochs,
        warmup_t=5,
        warmup_lr_init=1e-6,
        lr_min=1e-5,
    )
        
    # gradscaler
    scaler = amp.GradScaler(enabled=args.use_amp)
    
    train_result_save = Result()
    valid_result_save = Result()
    test_result_save  = Result()
    
    best_MAE = float('inf')
    
    print(f'Train loader length: {len(train_loader)}')
    print(f'Valid loader length: {len(valid_loader)}')
    print(f'Test  loader length: {len(test_loader)}')
    print(f'Training start for {args.epochs} epochs')

    # train
    for epoch in range(args.epochs):
        train_result_save.reset()
        valid_result_save.reset()
        test_result_save.reset()
        
        print(f'\nEpoch: {epoch + 1}')
        
        model.train()
        for idx, data in enumerate(tqdm(train_loader)):
            image = data['image'].to(device)
            ir_drop = [d.to(device) for d in data['ir_drop']]
            
            with amp.autocast(enabled=args.use_amp):
                pred, aux_pred = model(image)
                pred = reverse_normalize(pred, mean['ir_drop'], std['ir_drop'])
                aux_pred = reverse_normalize(aux_pred, mean['ir_drop'], std['ir_drop'])

                loss = criterion(pred, ir_drop) + aux_criterion(aux_pred, ir_drop)
                loss = loss / args.gradient_accumulation
                
            scaler.scale(loss).backward()

            if (idx + 1) % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            result = train_result_save.update(pred, ir_drop, loss.item() * args.gradient_accumulation)
            
            if args.wandb:
                wandb.log({
                    'iteration': epoch * len(train_loader) + idx + 1,
                    'train Loss': result['loss'],
                    'train MSE': result['mse'],
                    'train MAE': result['mae'],
                    'train RMSE': result['rmse'],
                    'train Max Error': result['max'],
                    'train F1': result['f1_score'],
                    'train Recall': result['recall'],
                    'train Precision': result['precision'],
                })
        
        # evaluation
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                image = data['image'].to(device)
                ir_drop = [d.to(device) for d in data['ir_drop']]

                with amp.autocast(enabled=args.use_amp):
                    pred, aux_pred = model(image)
                    pred = reverse_normalize(pred, mean['ir_drop'], std['ir_drop'])
                    aux_pred = reverse_normalize(aux_pred, mean['ir_drop'], std['ir_drop'])

                    loss = criterion(pred, ir_drop) + aux_criterion(aux_pred, ir_drop)
                    
                valid_result_save.update(pred, ir_drop, loss.item())
            
            for idx, data in enumerate(test_loader):
                image = data['image'].to(device)
                ir_drop = [d.to(device) for d in data['ir_drop']]

                with amp.autocast(enabled=args.use_amp):
                    pred, aux_pred = model(image)
                    pred = reverse_normalize(pred, mean['ir_drop'], std['ir_drop'])
                    aux_pred = reverse_normalize(aux_pred, mean['ir_drop'], std['ir_drop'])

                    loss = criterion(pred, ir_drop) + aux_criterion(aux_pred, ir_drop)
                    
                test_result_save.update(pred, ir_drop, loss.item())
        
        # lr schedule
        scheduler.step(epoch + 1)
        
        # get result
        print(f'-------------------- Train --------------------')
        train_result = train_result_save.average()
        train_result_save.print()
        
        print(f'-------------------- Valid --------------------')
        valid_result = valid_result_save.average()
        valid_result_save.print()
        
        print(f'-------------------- Test  --------------------')
        test_result  = test_result_save.average()
        test_result_save.print()
        
        # wandb log
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                
                'epoch train Loss': train_result['loss'],
                'epoch train MSE': train_result['mse'],
                'epoch train MAE': train_result['mae'],
                'epoch train RMSE': train_result['rmse'],
                'epoch train Max Error': train_result['max'],
                'epoch train F1': train_result['f1_score'],
                'epoch train Recall': train_result['recall'],
                'epoch train Precision': train_result['precision'],
                
                'epoch valid Loss': valid_result['loss'],
                'epoch valid MSE': valid_result['mse'],
                'epoch valid MAE': valid_result['mae'],
                'epoch valid RMSE': valid_result['rmse'],
                'epoch valid Max Error': valid_result['max'],
                'epoch valid F1': valid_result['f1_score'],
                'epoch valid Recall': valid_result['recall'],
                'epoch valid Precision': valid_result['precision'],
                
                'epoch test Loss': test_result['loss'],
                'epoch test MSE': test_result['mse'],
                'epoch test MAE': test_result['mae'],
                'epoch test RMSE': test_result['rmse'],
                'epoch test Max Error': test_result['max'],
                'epoch test F1': test_result['f1_score'],
                'epoch test Recall': test_result['recall'],
                'epoch test Precision': test_result['precision'],
            })

        # save best model
        if valid_result['mae'] < best_MAE:
            best_MAE = valid_result['mae']
        
            if args.save_best_model:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                torch.save(model.state_dict(), f'{args.save_dir}/checkpoint.pth')
                print(f'\nBest model saved - epoch: {epoch + 1} - MAE: {best_MAE}')

    return model

def evaluate(args, model, valid_loader, test_loader, mean, std, device):
    valid_result_save = Result()
    test_result_save  = Result()

    criterion = RMSELoss()
    aux_criterion = DiceLoss(mean=mean['ir_drop'], std=std['ir_drop'])

    # evaluation
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            image = data['image'].to(device)
            ir_drop = [d.to(device) for d in data['ir_drop']]

            with amp.autocast(enabled=args.use_amp):
                pred, aux_pred = model(image)
                pred = reverse_normalize(pred, mean['ir_drop'], std['ir_drop'])
                aux_pred = reverse_normalize(aux_pred, mean['ir_drop'], std['ir_drop'])

                loss = criterion(pred, ir_drop) + aux_criterion(aux_pred, ir_drop)

            valid_result_save.update(pred, ir_drop, loss.item())
        
        for idx, data in enumerate(test_loader):
            image = data['image'].to(device)
            ir_drop = [d.to(device) for d in data['ir_drop']]

            with amp.autocast(enabled=args.use_amp):
                pred, aux_pred = model(image)
                pred = reverse_normalize(pred, mean['ir_drop'], std['ir_drop'])
                aux_pred = reverse_normalize(aux_pred, mean['ir_drop'], std['ir_drop'])

                loss = criterion(pred, ir_drop) + aux_criterion(aux_pred, ir_drop)

            test_result_save.update(pred, ir_drop, loss.item())

            
            threshold = 0.9
            dbu = 2000  # 1 픽셀 = 1μm, → DBU 기준 변환

            indices_list = get_exceeding_indices(pred, threshold=threshold)

            for i, indices in enumerate(indices_list):
                print(f"[Test] Sample {idx * len(image) + i}: {len(indices)} points exceed {threshold*100:.1f}% of max")
                sample_idx = idx * len(image) + i
                if len(indices) > 0:
                    pred_shape = pred.shape[-2:]  # (H, W)

                    # 1️⃣ 픽셀 좌표만 저장 (.txt)
                    txt_filename = f"cfirstnet_output_{sample_idx}.txt"
                    with open(txt_filename, "w") as log_file:
                        for j, (y, x) in enumerate(indices):
                            log_file.write(f"Position {j}: tensor_y={y.item()}, tensor_x={x.item()}\n")
                            if j < 5:
                                print(f"  Position {j}: tensor_y={y.item()}, tensor_x={x.item()}")


                    # 2️⃣ 변환 함수 정의 (DBU 적용한 XMODEL 노드)
                    def position_to_xmodel(lines, dbu=2000):
                        output_lines = []
                        idx = 0
                        for line in lines:
                            if "tensor_y=" in line and "tensor_x=" in line:
                                try:
                                    parts = line.strip().replace("Position: ", "").split(", ")
                                    py = int(parts[0].split("=")[1])
                                    px = int(parts[1].split("=")[1])
                                    py_dbu = py * dbu
                                    px_dbu = px * dbu

                                    # 예시 노드명: n1_m1_440000_312000
                                    nodename = f"n1_m1_{px_dbu}_{py_dbu}"
                                    output_lines.append(f"Ihotspot{idx} {nodename} 0 DC 3mA")
                                    idx += 1
                                except Exception as e:
                                    print("❌ Parse error:", line, e)
                        return output_lines



                    # 3️⃣ 변환 수행
                    with open(txt_filename) as f:
                        lines = f.readlines()

                    xmodel_snippets = position_to_xmodel(lines)

                    # 4️⃣ .sp 저장
                    sp_filename = f"hotspots_xmodel_{sample_idx}.sp"
                    with open(sp_filename, "w") as f:
                        for line in xmodel_snippets:
                            f.write(line + "\n")

                    print(f"✅ XMODEL netlist 파일 생성 완료: {sp_filename}")



    # get result
    print(f'-------------------- Valid --------------------')
    valid_result_save.print()
    
    print(f'-------------------- Test  --------------------')
    test_result_save.print()
