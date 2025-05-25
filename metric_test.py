from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from model.metric import *

from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import xavier_init_jittor
from model_jittor.metric_jittor import *

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

random_sample=np.random.randint(0, 256, size=(64,3,256,256))
random_sample=random_sample.astype('float32')/255.0

ROC=ROCMetric(1,10)
mIoU = mIoU(1)
PD_FA = PD_FA(1,10)
ROC_jt=ROCMetric_jt(10)
mIoU_jt=mIoU_jt()
PD_FA_jt=PD_FA_jt(10)

with open("result_jt\\match.log", 'a') as file:
    file.write("metric test:\n")

for i in range(64):
    batch_data=random_sample[i:i+1,:,:,:]
    label=np.random.randint(0, 2, size=(1,1,256,256))
    #predict=np.random.randint(0, 2, size=(1,1,256,256))
    #pytorch forward
    torch_data=torch.from_numpy(batch_data)
    torch_data=torch_data.cuda()
    label_torch = torch.from_numpy(label).cuda()
    predict_torch=model(torch_data)
    predict_torch=predict_torch[3]
    #predict_torch=torch.from_numpy(predict).cuda()
    ROC.update(predict_torch, label_torch)
    mIoU.update(predict_torch, label_torch)
    PD_FA.update(predict_torch, label_torch)
    #jittor forward
    jt_data=jt.array(batch_data)
    label_jittor=jt.array(label)
    predict_jittor=model_jittor(jt_data)
    predict_jittor=predict_jittor[3]
    #predict_jittor=jt.array(predict)
    ROC_jt.update(predict_jittor, label_jittor)
    mIoU_jt.update(predict_jittor, label_jittor)
    PD_FA_jt.update(predict_jittor, label_jittor)
#eval
ture_positive_rate, false_positive_rate, recall, precision= ROC.get()
ture_positive_rate_jt, false_positive_rate_jt, recall_jt, precision_jt= ROC_jt.get()
pixAcc, mIoU=mIoU.get()
pixAcc_jt,mIoU_jt=mIoU_jt.get()
FA, PD = PD_FA.get(64)
FA_jt, PD_jt = PD_FA_jt.get(64)
#print("error:")
#print(f"ture_positive_rate={ture_positive_rate-ture_positive_rate_jt}, {ture_positive_rate_jt}")
#print(f"false_positive_rate={false_positive_rate-false_positive_rate_jt}, {false_positive_rate_jt}")
#print(f"recall={recall-recall_jt}")
#print(f"precision={precision-precision_jt}")
#print(f"pixAcc={pixAcc.item()-pixAcc_jt}")
#print(f"mIoU={mIoU.item()-mIoU_jt[0]}")
#print(f"FA={FA-FA_jt}")
#print(f"PD={PD-PD_jt}")
with open("result_jt\\match.log", 'a') as file:
    file.write(f"ture_positive_rate error={ture_positive_rate-ture_positive_rate_jt}\n")
    file.write(f"false_positive_rate error={false_positive_rate-false_positive_rate_jt}\n")
    file.write(f"recall error={recall-recall_jt}\n")
    file.write(f"precision error={precision-precision_jt}\n")
    file.write(f"pixAcc error={pixAcc.item()-pixAcc_jt}\n")
    file.write(f"mIoU error={mIoU.item()-mIoU_jt}\n")
    file.write(f"FA error={FA-FA_jt}\n")
    file.write(f"PD error={PD-PD_jt}\n")