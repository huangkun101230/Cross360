import os
import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F

from Networks.network import OursNetwork

import datasets
from torch.utils.data import DataLoader

import wandb

from utils.utils import EarlyStopping
from utils.losses import *
from utils.loss_gradient import Gradient_Loss
from Metrics.depth_metrics import Depth_Evaluator

torch.backends.cudnn.enabled = False
np.set_printoptions(threshold=np.inf)


def dense_train(model, device, train_loader, optimizer, depth_gradient_loss, depth_MSE_loss, epoch, enable_wandb, scaler=None):
    model.train()
    model.to(device)

    depth_gradient_loss_criterion = depth_gradient_loss.to(device)
    depth_MSE_loss_criterion = depth_MSE_loss.to(device)

    pbar = tqdm.tqdm(train_loader)
    pbar.set_description("Training Epoch_{}".format(epoch))

    for batch_idx, train_sample in enumerate(pbar):
        ori_rgb = train_sample["ori_rgb"].to(device)
        depth = train_sample["gt_depth"].to(device)
        mask = train_sample["depth_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(ori_rgb)

        pred_depth_multiscale = outputs["pred_multiscale_depth"]
        pred_depth_multiscale = [torch.median(depth) / torch.median(pred_depth_multiscale[i]* mask) * (pred_depth_multiscale[i]* mask) for i in range(len(pred_depth_multiscale))]

        depth_mse_loss_outcome = sum([depth_MSE_loss_criterion(pred_depth_multiscale[i], depth) for i in range(len(pred_depth_multiscale))])
        depth_gradient_loss_outcome = sum([depth_gradient_loss_criterion(pred_depth_multiscale[i], depth) for i in range(len(pred_depth_multiscale))])

        loss = depth_mse_loss_outcome+depth_gradient_loss_outcome

        optimizer.zero_grad()

        # print("depth_mse_loss_outcome",depth_mse_loss_outcome)
        # print("depth_gradient_loss_outcome",depth_gradient_loss_outcome)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if enable_wandb:
            wandb.log({"Normal Training loss": loss.item()})


def dense_val(model, device, vali_loader, epoch, depth_evaluator, enable_wandb):
    model.eval()
    model.to(device)

    depth_evaluator.reset_eval_metrics()


    pbar = tqdm.tqdm(vali_loader)
    pbar.set_description("Validating Epoch_{}".format(epoch))
    loss = 0
    with torch.no_grad():
        for batch_idx, val_sample in enumerate(pbar):
            ori_rgb = val_sample["ori_rgb"].to(device)
            depth = val_sample["gt_depth"].to(device)
            mask = val_sample["depth_mask"].to(device)

            outputs = model(ori_rgb)

            pred_depth = outputs["pred_depth"] * mask
            pred_depth = torch.median(depth) / torch.median(pred_depth) * pred_depth
            depth_evaluator.compute_eval_metrics(depth.detach().cpu(), pred_depth.detach().cpu())
        
        depth_evaluator.print()
        depthloss = depth_evaluator.get_combined_err()

        if enable_wandb:
            wandb.log({"Depth Validation loss": depthloss})

    return depthloss

    
def trainer(settings):
    folder = settings.saving_folder
    model_path = './'+folder+'/models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    enable_wandb = settings.enable_wandb
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    if enable_wandb:
        wandb.init(project = "Cross360")
        wandbconfig = wandb.config


    print("use_cuda %s:" % str(settings.using_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.using_gpu)
    device = torch.device(settings.using_gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # set the current device to the new GPU

    # data
    datasets_dict = {"3d60": datasets.Dataset3D60,
                     "structured3d": datasets.Structured3d
                     }
    dataset = datasets_dict[settings.dataset]
    fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

    train_file_list = fpath.format(settings.dataset, "train")
    val_file_list = fpath.format(settings.dataset, "val")

    train_dataset = dataset(train_file_list, settings.height, settings.width, is_training=True)
    train_loader = DataLoader(train_dataset, settings.batch_size, shuffle=True,
                                   num_workers=settings.num_workers, pin_memory=True, drop_last=True)

    val_dataset = dataset(val_file_list, settings.height, settings.width, is_training=False)
    val_loader = DataLoader(val_dataset, settings.batch_size, shuffle=False,
                                 num_workers=settings.num_workers, pin_memory=True, drop_last=True)

    model = OursNetwork()

    model.to(device)

    if enable_wandb:
        wandb.watch(model)

    depth_evaluator = Depth_Evaluator()
    depth_MSE_loss = MSE()
    depth_gradient_loss = Gradient_Loss(device)


    if settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    elif settings.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum)

    early_stopping = EarlyStopping(settings.es)
    scaler = torch.cuda.amp.GradScaler() if settings.amp else None

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings.lr_decay, gamma=0.5, last_epoch=-1,verbose=True) #余弦退火衰减学习率

    start_epoch = 1
    for epoch in range(start_epoch, settings.epochs + 1):
        print('{}Network {}'.format('='*10, '='*10))
        torch.cuda.empty_cache()
        print("Current learning rate:",optimizer.state_dict()['param_groups'][0]['lr'])
        dense_train(model, device, train_loader, optimizer, depth_gradient_loss, depth_MSE_loss, epoch, enable_wandb, scaler=scaler)

        vali_epoch_loss = dense_val(model, device, val_loader, epoch, depth_evaluator, enable_wandb)
        early_stopping(vali_epoch_loss)
        scheduler.step()  

        if early_stopping.current_is_best:
            print('Saving Best model at: %s epoch!' % str(epoch))
            f = open("%slog.txt" % model_path, "w")
            f.write("Saving %s's model!\n" % (str(epoch)))
            f.close()
            all_state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            if settings.amp:
                all_state.update({'scaler':scaler.state_dict()})
            torch.save(all_state, '{}best_model.pkl'.format(model_path))

        if early_stopping.early_stop:
            print('Early Stopped at: %s epoch!' % str(epoch))
            print("Current learning rate:",optimizer.state_dict()['param_groups'][0]['lr'])
            all_state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            if settings.amp:
                all_state.update({'scaler':scaler.state_dict()})
            torch.save(all_state, '{}early_stop_model_{}.pkl'.format(model_path, epoch))
            break

    if enable_wandb:
        wandb.finish()    