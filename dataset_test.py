import torch
import numpy as np
from torchvision import transforms
import jittor as jt
import random
from torch.utils.data import DataLoader

from model.utils import TrainSetLoader,TestSetLoader
from model_jittor.utils_jittor import train_dataset,test_dataset
from model.load_param_data import load_dataset

if __name__ == '__main__':
    root="dataset"
    dataset="NUDT-SIRST"
    split="50_50"
    train_idx,test_idx,test_txt=load_dataset(root,dataset,split)
    dir="dataset\\NUDT-SIRST"

    #pytorch
    torch_test_img=[]
    torch_test_mask=[]

    input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])  #定义预处理操作
    trainset= TrainSetLoader(dir,img_id=train_idx,base_size=256,crop_size=256,transform=input_transform,suffix='.png')
    testset= TestSetLoader(dir,img_id=test_idx,base_size=256, crop_size=256, transform=input_transform,suffix='.png')
    train_data = DataLoader(dataset=trainset, batch_size=4, shuffle=False, num_workers=0,drop_last=True)
    test_data  = DataLoader(dataset=testset,  batch_size=4, shuffle=False, num_workers=0,drop_last=False)

    with open("result_jt\\match.log", 'a') as file:
        file.write("dataset test:\n")

    for i,(data, labels) in enumerate(train_data):
        data=data.cuda()
        labels=labels.cuda()
        torch_test_img.append(data.cpu().detach().numpy())
        torch_test_mask.append(labels.cpu().detach().numpy())
        if(i==4):
            break
    for i,(data, labels) in enumerate(test_data):
        data=data.cuda()
        labels=labels.cuda()
        torch_test_img.append(data.cpu().detach().numpy())
        torch_test_mask.append(labels.cpu().detach().numpy())
        if(i==4):
            break

    #jittor
    jittor_test_img=[]
    jittor_test_mask=[]
    jt.flags.use_cuda = True

    trainset_jt=train_dataset(dir,train_idx=train_idx,base_size=256,crop_size=256,suffix='.png')
    testset_jt=test_dataset(dir,test_idx=test_idx,base_size=256,crop_size=256,suffix='.png')
    train_data_jt=jt.dataset.DataLoader(trainset_jt, batch_size=4, shuffle=False, num_workers=0,drop_last=True)
    test_data_jt=jt.dataset.DataLoader(testset_jt,  batch_size=4, shuffle=False, num_workers=0,drop_last=False)

    i=0
    for data, labels in train_data_jt:
        jittor_test_img.append(data.numpy())
        jittor_test_mask.append(labels.numpy())
        if(i==4):
            break
        i=i+1

    i=0
    for data, labels in test_data_jt:
        jittor_test_img.append(data.numpy())
        jittor_test_mask.append(labels.numpy())
        if(i==4):
            break
        i=i+1

    for i in range(10):
        err=np.mean(torch_test_img[i]-jittor_test_img[i])
        #print(f"batch{i}: image error={err}")
        err2 = np.mean(torch_test_mask[i] - jittor_test_mask[i])
        #print(f"batch{i}: mask error={err2}")
        with open("result_jt\\match.log", 'a') as file:
            file.write(f"batch{i}: image error={err}\n")
            file.write(f"batch{i}: mask error={err2}\n")