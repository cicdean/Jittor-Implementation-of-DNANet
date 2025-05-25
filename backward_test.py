from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from model.loss import *
from torchvision import transforms
from torch.utils.data import DataLoader

from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import *
from model_jittor.loss_jittor import *

from model.utils import TrainSetLoader,TestSetLoader
from model_jittor.utils_jittor import train_dataset,test_dataset
from model.load_param_data import load_dataset

import numpy as np
import torch
import jittor as jt
from tqdm import tqdm

if __name__ == '__main__':
    nb_filter = [16, 32, 64, 128, 256]
    num_blocks = [2, 2, 2, 2]
    jt.flags.use_cuda = 1

    with open("result_jt\\match.log", 'a') as file:
        file.write("backward test:\n")

    #model init
    model=DNANet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
    model_jittor=DNANet_jittor(classes=1,in_channels=3,block=Res_CBAM_block_jittor,num_blocks=num_blocks,nb_filter=nb_filter,deep_supervision=True)
    model_jittor.apply(xavier_init_jittor)
    jittor_param = model_jittor.state_dict(to="torch")
    print(jittor_param.keys())
    torch_param=model.state_dict()
    print(torch_param.keys())
    model.load_state_dict(jittor_param)
    model.cuda()

    #optimizer
    lr=0.05
    epchos=1500
    min_lr=1e-5
    torch_p=filter(lambda p: p.requires_grad, model.parameters())
    optimizer_torch = torch.optim.Adagrad(torch_p, lr=lr)
    scheduler_torch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_torch, T_max=epchos, eta_min=min_lr)
    #scheduler_torch.step()
    jt_p=filter(lambda p: p.requires_grad, model_jittor.parameters())
    optimizer_jt=Adagrad(list(jt_p), lr=lr, eps=1e-10)
    scheduler_jt=jt.lr_scheduler.CosineAnnealingLR(optimizer_jt, T_max=epchos, eta_min=min_lr)
    scheduler_jt.step()
    #model.eval()
    #model_jittor.eval()
    model.train()
    model_jittor.train()

    root = "dataset"
    dataset = "NUDT-SIRST"
    split = "50_50"
    train_idx, test_idx, test_txt = load_dataset(root, dataset, split)
    dir = "dataset\\NUDT-SIRST"

    # pytorch dataset
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])  # 定义预处理操作
    trainset = TrainSetLoader(dir, img_id=train_idx, base_size=256, crop_size=256, transform=input_transform,
                              suffix='.png')
    testset = TestSetLoader(dir, img_id=test_idx, base_size=256, crop_size=256, transform=input_transform,
                            suffix='.png')
    train_data = DataLoader(dataset=trainset, batch_size=4, shuffle=False, num_workers=0, drop_last=True)
    test_data = DataLoader(dataset=testset, batch_size=4, num_workers=0, drop_last=False)

    #jittor dataset
    trainset_jt = train_dataset(dir, train_idx=train_idx, base_size=256, crop_size=256, suffix='.png')
    testset_jt = test_dataset(dir, test_idx=test_idx, base_size=256, crop_size=256, suffix='.png')
    train_data_jt = jt.dataset.DataLoader(trainset_jt, batch_size=4, shuffle=False, num_workers=0, drop_last=True)
    test_data_jt = jt.dataset.DataLoader(testset_jt, batch_size=4, num_workers=0, drop_last=False)

    torch.cuda.empty_cache()
    torch_loss=[]
    torch_lr=[]
    #torch_training
    i=0
    for data, labels in train_data:
        data = data.cuda()
        labels = labels.cuda()
        preds = model(data)
        loss = 0
        for pred in preds:
            loss += SoftIoULoss(pred, labels)
        loss /= len(preds)
        optimizer_torch.zero_grad()
        loss.backward()
        '''for name, param in model.named_parameters():
            print(f"Parameter: {name}")
            print(f"Gradient: {param.grad}")'''
        optimizer_torch.step()
        torch_loss.append(loss.item())
        torch_lr.append(scheduler_torch.get_last_lr())
        if(i==5):
            break
        i+=1

    jittor_loss=[]
    torch.cuda.empty_cache()
    jittor_lr=[]
    #jittor training
    i=0
    for data, labels in train_data_jt:
        data=jt.array(data)
        #data=jt.misc.to(data,'cuda')
        label=jt.array(labels)
        #label = jt.misc.to(label, 'cuda')
        preds = model_jittor(data)
        loss = 0
        for pred in preds:
            loss += SoftIoULoss_jt(pred, labels)
        loss /= len(preds)
        #optimizer_jt.zero_grad()
        #optimizer_jt.backward(loss)
        '''for param in optimizer_jt.param_groups[0]['params']:
            grad = param.opt_grad(optimizer_jt)
            print(f"Gradient: {jt.tolist(grad)}")'''
        #optimizer_jt.step()
        optimizer_jt.step(loss)
        jittor_loss.append(loss.item())
        jittor_lr.append(scheduler_jt.optimizer.lr)
        if (i == 5):
            break
        i += 1

    jittor_loss=np.array(jittor_loss)
    jittor_lr=np.array(jittor_lr)
    torch_loss=np.array(torch_loss)
    torch_lr=np.array(torch_lr).reshape((1,6))
    #print(torch_loss)
    #print(jittor_loss)
    error_loss=torch_loss-jittor_loss
    #print(f"error_loss={error_loss}")
    #print(torch_lr)
    #print(jittor_lr)
    error_lr=torch_lr-jittor_lr
    #print(f"error_lr={error_lr}")
    with open("result_jt\\match.log", 'a') as file:
        file.write(f"error_loss={error_loss}\n")
        file.write(f"error_lr={error_lr}\n")