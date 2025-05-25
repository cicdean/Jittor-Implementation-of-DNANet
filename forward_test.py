from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import xavier_init_jittor

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
random_sample=random_sample.astype('float32')/255.0

with open("result_jt\\match.log", 'a') as file:
    file.write("forward test:\n")

for i in range(16//4):
    batch_data=random_sample[i*4:i*4+4,:,:,:]
    #pytorch forward
    torch_data=torch.from_numpy(batch_data)
    torch_data=torch_data.cuda()
    predict_torch=model(torch_data)
    r_torch = np.zeros((4, 0, 256, 256))
    for j in range(len(predict_torch)):
        r_torch=np.concatenate((r_torch,predict_torch[i].cpu().detach().numpy()),axis=1)
    #jittor forward
    jt_data=jt.array(batch_data)
    predict_jittor=model_jittor(jt_data)
    r_jittor = np.zeros((4, 0, 256, 256))
    for j in range(4):
        r_jittor = np.concatenate((r_jittor, predict_jittor[i].numpy()), axis=1)
    #eval
    err = np.mean(np.abs(r_torch - r_jittor))
    #print("pytorch result:")
    #print(r_torch)
    #print("jittor_result:")
    #print(r_jittor)
    #print(f"batch{i}: mean error: {err}")
    with open("result_jt\\match.log", 'a') as file:
        file.write(f"batch{i}: mean error: {err}" + '\n')
