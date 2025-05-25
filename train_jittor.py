import os
from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import *
from model_jittor.loss_jittor import *
from model_jittor.load import *
from model_jittor.metric_jittor import *
from tqdm import tqdm
jt.flags.use_cuda = 1

epchos=450   #450
channel_size=3
backbone="resnet_18"
deep_supervision=True
root = "dataset"
dataset = "NUDT-SIRST"
split = "50_50"
lr=0.05
min_lr=1e-5

class Trainer():
    def __init__(self):
        #evaluation methods
        self.ROCMetric=ROCMetric_jt(10)
        self.mIoU = mIoU_jt()
        self.PD_FA = PD_FA_jt(10)
        #save dir
        self.save_dir=makdir(dataset,True)
        save_train_log(self.save_dir,epchos,channel_size,backbone,deep_supervision,dataset,split,lr,min_lr)
        self.save_prefix = f"{dataset}_DNANet"
        #load data
        self.train_idx, self.test_idx, test_txt = load_dataset(root, dataset, split)
        self.dir=root+"\\"+dataset
        trainset_jt = train_dataset(self.dir, train_idx=self.train_idx, base_size=256, crop_size=256, suffix='.png')
        testset_jt = test_dataset(self.dir, test_idx=self.test_idx, base_size=256, crop_size=256, suffix='.png')
        self.train_data = jt.dataset.DataLoader(trainset_jt, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
        self.test_data = jt.dataset.DataLoader(testset_jt, batch_size=4, num_workers=0, drop_last=False)
        #initialize model
        nb_filter, num_blocks = load_param(channel_size, backbone)
        self.model=DNANet_jittor(classes=1,in_channels=3,block=Res_CBAM_block_jittor,num_blocks=num_blocks,nb_filter=nb_filter,deep_supervision=True)
        self.model.apply(xavier_init_jittor)
        #print(self.model.parameters())
        print("Model Initializing")
        #define optimizer
        jt_p = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = Adagrad(list(jt_p), lr=lr, eps=1e-10)
        #self.optimizer=jt.nn.Adam(self.model.parameters(), lr=0.05, eps=1e-8, betas=(0.9, 0.999))
        #self.scheduler = jt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epchos, eta_min=min_lr)
        #self.scheduler.step()
        #best evaluation result
        self.best_iou = 0
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.loss_list=[]

    def train(self,epoch):
        jt.clean_graph()
        jt.sync_all()
        jt.gc()
        losses = AverageMeter()
        tbar = tqdm(self.train_data)
        self.model.train()
        #i=0
        for data, labels in tbar:
            #data = jt.misc.to(data, 'cuda')
            #labels = jt.misc.to(labels, 'cuda')
            loss = 0
            if deep_supervision:
                preds = self.model(data)
                for pred in preds:
                    loss += SoftIoULoss_jt(pred, labels)
                loss /= len(preds)
            else:
                pred = self.model(data)
                loss = SoftIoULoss_jt(pred, labels)
            self.optimizer.step(loss)
            #self.scheduler.step()
            #self.optimizer.zero_grad()
            #self.optimizer.backward(loss)
            #self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description(f"Epoch {epoch}, training loss {losses.avg:.4f}")
            #i+=1
            #if i==50:
                #break
        self.train_loss = losses.avg

    def test(self,epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.ROCMetric.reset()
        losses = AverageMeter()
        with jt.no_grad():
            #i = 0
            for data, labels in tbar:
                loss = 0
                if deep_supervision:
                    preds = self.model(data)
                    for pred in preds:
                        loss += SoftIoULoss_jt(pred, labels)
                    loss /= len(preds)
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss_jt(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROCMetric.update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROCMetric.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description(f"Epoch {epoch}, test loss {losses.avg:.4f}, mean_IoU: {mean_IOU:4f}")
                #i += 1
                #if i == 50:
                    #break
            test_loss = losses.avg
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        print(self.best_iou)
        with open(f"result_jt\\{self.save_dir}\\loss.txt", 'a') as file:
            file.write(str(test_loss) + '\n')

    def save_loss(self):
        with open(f"result_jt\\{self.save_dir}\\loss.txt", 'w') as file:
            for item in self.loss_list:
                file.write(str(item) + '\n')

if __name__ == "__main__":
    trainer = Trainer()
    for epoch in range(epchos):
        trainer.train(epoch)
        trainer.test(epoch)
    #trainer.save_loss()