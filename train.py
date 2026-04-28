import os, argparse, glob, time, cv2, pdb, torch, models
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random as rn
from scipy.io import savemat, loadmat
from utils import *
from dataset import *
from math import acos, degrees
from tensorboardX import SummaryWriter 
from trainOps import *
from tqdm import tqdm

from models.hssdct import HyDCFN

torch.backends.cudnn.benchmark=True

# ============================================================
# HSSDCT 训练脚本 (Training Script)
# 
# Paper: "HSSDCT: Factorized Spatial-Spectral Correlation for
#        Hyperspectral Image Fusion" (arXiv:2602.00490)
#
# 本脚本实现了HSSDCT模型的完整训练流程，包括：
#   1. 数据加载（支持多种数据集模式）
#   2. 模型初始化与断点恢复
#   3. 复合损失函数训练（论文公式5）
#   4. 验证与评估（PSNR/SAM/RMSE/ERGAS）
#
# 【论文实验设置】(Section 3.1 Implementation Details):
#   - 框架: PyTorch + NVIDIA RTX 3090
#   - Batch size: 4, Epochs: 600
#   - 优化器: Adam, 初始学习率: 0.0001 (此处使用0.000055)
#   - 学习率调度: Cosine Annealing
#   - HDRTB窗口大小: {4, 8, 16, 16}
#   - 数据集: AVIRIS (2078张HR-HSI图像，训练1678/验证200/测试200)
#   - 输入尺寸: LR-HSI=64×64×172, HR-MSI=256×256×Mm(Mm=4或6)
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    # ==================== 基础训练参数 ====================
    # 【论文对应】默认值可能与论文有差异以适配不同实验场景
    parser.add_argument('--SEED', type=int, default=1029)           # 随机种子，保证可复现性
    parser.add_argument('--batch_size', type=int, default=6)        # 批大小（论文中为4）

    parser.add_argument('--epochs', type=int, default=1000)         # 训练轮数（论文中为600）
    parser.add_argument('--lr_scheduler', type=str, default="cosine") # 学习率调度策略：cosine/step
    parser.add_argument('--resume_ind', type=int, default=0)        # 断点恢复的起始epoch
    parser.add_argument('--resume_ckpt', type=str, default="")      # 断点恢复的模型路径
    parser.add_argument('--snr', type=int, default=35)              # AWGN信噪比(dB)，用于噪声增强训练
    
    parser.add_argument('--lr', type=float, default=0.000055)       # 初始学习率（论文中为0.0001）
    parser.add_argument('--step_size', type=int, default=200)       # StepLR的学习率衰减步长
    parser.add_argument('--workers', type=int, default=4)           # 数据加载线程数
    parser.add_argument('--eval_step', type=int, default=1)         # 验证频率（每N个epoch验证一次）
    parser.add_argument('--finetuning_step', type=int, default=200, help='Works only if the mixed_align_opt is on')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate, 0 means training without weight decay')
    
    
    ## Data generator configuration（数据生成配置，对应论文数据集设置）
    parser.add_argument('--crop_size', type=int, default=128)       # 训练时的裁剪大小
    parser.add_argument('--image_size', type=int, default=256)      # 目标输出图像大小（HR-HSI空间分辨率）
    parser.add_argument('--bands', type=int, default=172)           # 高光谱波段数（AVIRIS数据集：172个有效波段）
    parser.add_argument('--msi_bands', type=int, default=4)         # 多光谱波段数（论文实验使用4或6）
    parser.add_argument('--hsi_bands', type=int, default=172)       # 高光谱输入波段数
    parser.add_argument('--mis_pix', type=int, default=0)           # 错位像素数（用于错位融合实验）
    parser.add_argument('--mixed_align_opt', type=int, default=0)   # 混合对齐优化选项
    parser.add_argument('--joint_loss', type=int, default=1)        # 是否使用联合损失（1=是，对应论文公式5）
    parser.add_argument('--gc', type=int, default=32)               # HDRTB中的growth channel数
    
    # Network architecture configuration（网络架构配置）
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=172, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    parser.add_argument('--out_nc', type=int, default=172, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    parser.add_argument('--nf', type=int, default=96, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others（其他参数）
    parser.add_argument("--root", type=str, default="/ssd4t/Fusion_data", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="./data_path/val.txt")   
    parser.add_argument("--train_file", type=str, default="./data_path/train.txt")   
    parser.add_argument("--prefix", type=str, default="ASTUDY_num2_BAND4_SWINY_SNR0")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args


def trainer(args):
    """【HSSDCT主训练流程】实现论文 Section 3 描述的完整训练与验证流程
    
    训练流程概览：
      1. 数据加载：根据network_mode选择数据集模式（配对/三元组/单模态）
      2. 模型初始化：创建HyDCFN模型，支持多GPU和断点恢复
      3. 优化器配置：Adam优化器 + CosineAnnealing学习率调度
      4. 训练循环：复合损失函数（论文公式5）+ 反向传播
      5. 验证评估：计算PSNR/SAM/RMSE/ERGAS指标（论文Table 1）
      6. 模型保存：保存best(SAM最优)和last(最新)检查点
    """
    # Print configuration - 打印所有配置参数便于实验复现
    print("#"*80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#"*80)
    
    flist = loadTxt(args.train_file)
    valfn = loadTxt(args.val_file)
    tlen = len(flist)
    print(f'#training samples is {tlen} and validation samples is {len(valfn)}')

    # ==================== 数据集选择（对应论文数据集设置） ====================
    # 论文使用AVIRIS数据集：2078张图像，训练1678/验证200/测试200
    # HR-MSI: 256×256×Mm, LR-HSI: 64×64×172, Mm=4或6
    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')  # 三元组模式 (LRHSI+HRMSI+CO)
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')  # 配对模式（HSSDCT主要使用的模式）
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')  # 单模态模式
    
    train_loader = torch.utils.data.DataLoader(dataset(flist, args), batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.workers)

    # ==================== 模型初始化与断点恢复 ====================
    model = HyDCFN(args).to(args.device)  # 创建HSSDCT模型（HyDCFN包装器）
    if args.gpus>1:
        model = torch.nn.DataParallel(model).to(args.device)  # 多GPU数据并行
    
    
    if args.resume_ind>0 or os.path.isfile(args.resume_ckpt):
        # 支持从检查点恢复训练
        if not os.path.isfile(args.resume_ckpt):
            args.resume_ckpt = os.path.join('checkpoint', args.prefix, 'best.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict = torch.load(args.resume_ckpt)  
        model.load_state_dict(state_dict)
        print(f'Loading the pretrained model from {args.resume_ckpt}')
    model.train()
    
    # ==================== 优化器配置（论文 Section 3.1） ====================
    # 论文使用 Adam 优化器，初始学习率 0.0001，Cosine Annealing 调度
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    if args.lr_scheduler=='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.resume_ind, eta_min=1e-10, last_epoch=-1)
        # Cosine Annealing: 学习率按余弦曲线从lr衰减到eta_min（论文推荐）
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
        
    L1Loss = torch.nn.SmoothL1Loss()  # 平滑L1损失（即论文公式5中的 L_L1 像素保真度损失）

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(f'checkpoint/{args.prefix}'):
        os.mkdir(f'checkpoint/{args.prefix}')
    if not os.path.isdir('Rec'):
        os.mkdir('Rec')    
    if not os.path.isdir(f'Rec/{args.prefix}'):
        os.mkdir(f'Rec/{args.prefix}')
    
    writer = SummaryWriter('log/%s_exp2' % (args.prefix))  # TensorBoard日志记录器
    
    resume_ind = args.resume_ind if args.resume_ind>0 else 0
    step = resume_ind
    best_sam = 99  # 最佳SAM值（越低越好），用于保存最优模型
    # ==================== 主训练循环 ====================
    for epoch in range(resume_ind, args.epochs): 
            
        ep_loss = 0.
        running_loss, running_sam, running_bws=[],[],[]  # 记录每个batch的损失用于epoch级统计
        t1 = time.perf_counter()
        optim_time = time.perf_counter()
        for batch_idx, (X) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # ---- 数据加载与GPU传输（支持多种数据集模式）----
            t2 = time.perf_counter()-t1
            if args.DEBUG:
                print(f'Sampling time: {t2} seconds')
            
            t1 = time.perf_counter()
            if args.network_mode==2:
                # 三元组模式: (HRMSI, LRHSI, CO/其他, GT, ...)
                x,x2,x3,y,_,_,_ = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                # 配对模式（HSSDCT主要使用）: (HRMSI, LRHSI, GT, ...)
                x,x2,y,_,_,_ = X  # x=HRMSI, x2=LRHSI, y=GT(HR-HSI)
                x2 = x2.cuda()
            elif args.network_mode==0:
                # 单模态模式: (Input, GT, ...)
                x,y,_,_,_ = X
                
            optimizer.zero_grad()
            x = x.cuda()   # HRMSI 或 Input
            y = y.cuda()   # Ground Truth HR-HSI
            if args.DEBUG:
                print(f'To cuda tensor time {time.perf_counter()-t1} seconds')
            
            t1 = time.perf_counter()
            
            # ---- 模型前向传播（HSSDCT融合重建）----
            if args.network_mode==2:
                decoded = model(x, LRHSI=x2, HRMSI=x3)
            elif args.network_mode==1:
                decoded = model(LRHSI=x2, HRMSI=x)  # HSSDCT标准调用：输入LR-HSI和HR-MSI，输出重建的HR-HSI
            elif args.network_mode==0:
                decoded = model(x, LRHSI=None, HRMSI=None)
                
            if args.DEBUG:
                print(f'model inference time{time.perf_counter()-t1} seconds')
                
            # ==================== 损失函数计算（论文公式5） ====================
            # 论文公式5: L_total = L_L1 + λ1*L_SAM + λ2*L_SWT
            # 其中: L_L1=像素保真度, L_SAM=光谱角制图(光谱保真度), L_SWT=小波变换(结构纹理一致性)
            # 论文中设置 λ1=λ2=0.01，此处使用 0.1 作为SAM和BWS损失的权重
            
            loss = L1Loss(decoded, y)           # L_L1: Smooth L1损失，保证像素级保真度
            loss2 = sam_loss(decoded, y)         # L_SAM: 光谱角(Spectral Angle Mapper)损失，保留光谱签名
            loss3 = BandWiseMSE(decoded, y)      # 频域/波段级MSE损失（对应论文中的L_SWT或其替代）
            
            # ---- 训练稳定性保护：NaN检测与学习率衰减 ----
            while torch.isnan(loss2) and scheduler.get_last_lr()[0]>1e-12:
                # 当SAM损失出现NaN时，强制降低学习率并恢复到上一checkpoint
                print('Force learning rate decay to', scheduler.get_last_lr()[0]/5)
                
                args.resume_ckpt = os.path.join('checkpoint', args.prefix, 'last.pth')
                state_dict = torch.load(args.resume_ckpt)  
                model.load_state_dict(state_dict)
                
                optimizer = optim.Adam(model.parameters(), lr=scheduler.get_last_lr()[0]/5)  
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
                args.joint_loss = 0  # 关闭联合损失，仅使用L1损失继续训练
                
                continue
                
            if torch.isnan(loss2):
                # 如果降低学习率后仍为NaN，终止训练
                print('It is unnecessary to optimize anymore...abort the process')
                raise
                            
            reg = torch.std(decoded)  # 正则化项：输出标准差（用于稳定训练）
            
            # ---- 联合损失计算（对应论文公式5） ----
            if args.joint_loss==1:
                # L_total = L_L1 + λ1*L_SAM + λ2*BWMSE
                # 此处 λ1=λ2=0.1（论文中使用0.01，此处根据实验调整）
                total_loss = loss + 0.1*loss2 + 0.1*loss3
            else:
                total_loss = loss  # 仅使用L1损失

            t1 = time.perf_counter()
            total_loss.backward()  # 反向传播计算梯度
            # 注意：论文中未提及梯度裁剪，此处直接使用原始梯度

            if args.DEBUG:
                print(f'Backward time {time.perf_counter()-t1} seconds')
            
            optimizer.step()  # 更新模型参数（Adam优化器）
            running_loss.append(loss.item())
            running_sam.append(loss2.item())
            running_bws.append(loss3.item())
            
        scheduler.step()
        optim_time = time.perf_counter()-optim_time

        # ==================== 验证评估阶段（对应论文 Table 1） ====================
        # 每eval_step个epoch进行一次验证，计算PSNR/SAM/RMSE/ERGAS指标
        # 这些指标是HSI融合领域标准评估指标，用于与SOTA方法对比
        if epoch% args.eval_step ==0:
            model.eval()  # 切换到评估模式（关闭Dropout等）
            with torch.no_grad():  # 验证时不计算梯度以节省显存
                rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
                # RMSE: 均方根误差(越低越好) - 像素级重建精度
                # SAM: 光谱角映射(越低越好) - 光谱保真度（论文主要优化目标）
                # PSNR: 峰值信噪比(越高越好) - 整体重建质量
                # ERGAS: 全局相对误差(越低越好) - 综合辐射质量
                
                ep = 0
                
                for ind2, X in tqdm(enumerate(val_loader), total=len(val_loader)):
                    # ---- 验证数据加载（与训练类似）----
                    if args.network_mode==2:
                        (vx, vx2, vx3, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                        vx3=vx3.cuda()
                    elif args.network_mode==1:
                        (vx, vx2, vy, vfn, maxv, minv) = X  # vx=HRMSI, vx2=LRHSI, vy=GT
                        vx2=vx2.cuda()
                    elif args.network_mode==0:
                        (vx, vy, vfn, maxv, minv) = X
                        
                    # 裁剪到目标图像大小并准备GT
                    vy = vy.cpu().numpy()
                    vy = vy[:,:,:args.image_size,:args.image_size]
                    vx=vx.cuda()
                    
                    # maxv/minv用于反归一化（数据预处理时做了[-1,1]或[0,1]的归一化）
                    maxv, minv = maxv.cpu().numpy(), minv.cpu().numpy()
                   
                    # ---- 模型推理（mode=1表示推理模式，不注入AWGN噪声）----
                    start_time = time.time()
                    if args.network_mode==2:
                        val_dec = model(vx, LRHSI=vx2, HRMSI=vx3, mode=1)
                    elif args.network_mode==1:
                        val_dec = model(LRHSI=vx2, HRMSI=vx, mode=1)
                    elif args.network_mode==0:
                        val_dec = model(vx, LRHSI=None, HRMSI=None, mode=1)
                    ep = ep+(time.time()-start_time)
                    
                    val_dec = val_dec.cpu().numpy()
                    
                    # ---- 计算各项评估指标并保存重建结果 ----
                    for predimg, gtimg,f, v1, v2 in zip(val_dec, vy, vfn, maxv, minv):
                        predimg = (predimg/2+0.5)   # 从[-1,1]反归一化到[0,1]
                        gtimg = (gtimg/2+0.5) 
                        
                        # 在归一化空间计算光谱相关指标（SAM/PSNR/ERGAS）
                        sams.append(sam2(predimg, gtimg))       # SAM: 光谱角映射
                        psnrs.append(psnr(predimg, gtimg))       # PSNR: 峰值信噪比
                        ergas.append(ERGAS(predimg, gtimg))      # ERGAS: 全局相对误差
                        
                        # 在原始数值空间计算RMSE（需要先恢复原始数值范围）
                        predimg = predimg * (v1-v2) + v2         # 反归一化到原始数据范围
                        gtimg = gtimg * (v1-v2) + v2
                        rmses.append(rmse(predimg, gtimg))        # RMSE: 均方根误差
                        
                        # 保存重建结果为.mat文件，用于后续分析和可视化
                        savemat(f'Rec/{args.prefix}/{os.path.basename(f)}.mat', {'pred':np.transpose(predimg,(1,2,0))})
                                        
                ep = ep / len(sams)
                # 打印完整的训练/验证指标（对应论文 Table 1 的评估格式）
                print('[epoch: %d] Loss: %.3f, Loss-SAM: %.3f, Loss-BWS: %.3f, val-rmse: %.3f, val-SAM: %.3f, val-PSNR: %.3f, val-ERGAS: %.3f, Inference time: %f ms, Optim time: %f, lr: %f' % (epoch, 100*np.mean(running_loss), np.mean(running_sam), np.mean(running_bws), np.mean(rmses), np.mean(sams), np.mean(psnrs), np.mean(ergas), ep*1000, optim_time,scheduler.get_last_lr()[0]))
                
                # ---- TensorBoard日志记录 ----
                # Log validation metrics to TensorBoard
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                writer.add_scalar('Validation PSNR', np.mean(psnrs), step)
                writer.add_scalar('Validation ERGAS', np.mean(ergas), step)
                writer.add_scalar('Validation RMSE/std', np.std(rmses), step)
                writer.add_scalar('Validation SAM/std', np.std(sams), step)
                writer.add_scalar('Validation PSNR/std', np.std(psnrs), step)
                writer.add_scalar('Validation ERGAS/std', np.std(ergas), step)

            model.train()  # 验证完成后切回训练模式
            
            # ---- 保存最优模型（以SAM为指标，SAM越低越好）----
            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(),  f'checkpoint/{args.prefix}/best.pth')
                # 保存当前最优检查点（用于论文报告的最佳结果）
                
            ep_loss += np.mean(running_loss)
            writer.add_scalar('Loss/Running loss', np.mean(running_loss), step)
            writer.add_scalar('Loss/Running SAM-loss', np.mean(running_sam), step)
            writer.add_scalar('Loss/Running Weighted-MSE', np.mean(running_bws), step)
                
            running_loss, running_sam, running_bws=[], [], []
            model.train()  # 确保切回训练模式（防止验证模式的dropout影响后续训练）
        
        # ---- 保存最新检查点（用于断点恢复）----
        torch.save(model.state_dict(), f'checkpoint/{args.prefix}/last.pth')
        
        step+=1


if __name__ == '__main__':
    # ==================== 程序入口 ====================
    args = parse_args()
    # 设置随机种子确保实验可复现性
    torch.manual_seed(args.SEED)
    rn.seed(args.SEED)
    np.random.seed(args.SEED)
    trainer(args)
