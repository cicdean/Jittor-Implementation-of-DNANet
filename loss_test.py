from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from model.loss import *

from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import xavier_init_jittor
from model_jittor.loss_jittor import *

import numpy as np
import torch
import jittor as jt

nb_filter = [16, 32, 64, 128, 256]
num_blocks = [2, 2, 2, 2]
jt.flags.use_cuda = True

model=DNANet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
model_jittor=DNANet_jittor(classes=1,in_channels=3,block=Res_CBAM_block_jittor,num_blocks=num_blocks,nb_filter=nb_filter,deep_supervision=True)
model_jittor.apply(xavier_init_jittor)
jittor_param = model_jittor.state_dict(to="torch")
print(jittor_param.keys())
torch_param=model.state_dict()
print(torch_param.keys())
model.load_state_dict(jittor_param)
model.cuda()
model.eval()
model_jittor.eval()

random_sample=np.random.randint(0, 256, size=(16,3,256,256))
random_mask=np.random.randint(0, 2, size=(16,1,256,256))
random_sample=random_sample.astype('float32')/255.0

with open("result_jt\\match.log", 'a') as file:
    file.write("loss test:\n")

for i in range(16//4):
    batch_data=random_sample[i*4:i*4+4,:,:,:]
    mask=random_mask[i*4:i*4+4,:,:,:]
    #pytorch forward
    torch_data=torch.from_numpy(batch_data)
    torch_data=torch_data.cuda()
    predict_torch=model(torch_data)
    torch_mask=torch.from_numpy(mask).cuda()
    loss_torch=0
    for j in range(len(predict_torch)):
        loss_torch+=SoftIoULoss(predict_torch[j],torch_mask)
    #jittor forward
    jt_data=jt.array(batch_data)
    jt_mask=jt.array(mask)
    predict_jittor=model_jittor(jt_data)
    loss_jt=0
    for j in range(len(predict_jittor)):
        loss_jt+=SoftIoULoss_jt(predict_jittor[j],jt_mask)
    #eval
    err = np.mean(np.abs(loss_torch.item() - loss_jt.item()))
    #print("pytorch result:")
    #print(r_torch)
    #print("jittor_result:")
    #print(r_jittor)
    #print(f"batch{i}: mean error: {err}")
    with open("result_jt\\match.log", 'a') as file:
        file.write(f"batch{i}: mean error: {err}\n")