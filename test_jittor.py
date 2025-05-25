from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import *
from model_jittor.loss_jittor import *
from model_jittor.load import *
from model_jittor.metric_jittor import *
from tqdm import tqdm
import scipy.io as scio

channel_size=3
backbone="resnet_18"
deep_supervision=True
root = "D:\\Desktop\\Jittor-Implementation-of-DNANet\\dataset"
dataset = "NUDT-SIRST"
split = "50_50"
jt.flags.use_cuda = 1
lr=0.05
min_lr=1e-5
model_dir="NUDT-SIRST_DNANet_24_05_2025_20_16_48_wDS\\mIoU__NUDT-SIRST_DNANet_epoch.pkl"
st_model="NUDT-SIRST_DNANet_24_05_2025_20_16_48_wDS" #"NUDT-SIRST_DNANet_23_05_2025_18_04_56_wDS"
epochs=450

if __name__ == "__main__":
    ROC = ROCMetric_jt(10)
    PD_FA = PD_FA_jt(10)
    mIoU = mIoU_jt()
    save_prefix = f"{dataset}_DNANet"

    train_idx, test_idx, test_txt = load_dataset(root, dataset, split)
    dir = root + "\\" + dataset
    testset_jt = test_dataset(dir, test_idx=test_idx, base_size=256, crop_size=256, suffix='.png')
    test_data = jt.dataset.DataLoader(testset_jt, batch_size=1, num_workers=0, drop_last=False)

    nb_filter, num_blocks = load_param(channel_size, backbone)
    model = DNANet_jittor(classes=1, in_channels=3, block=Res_CBAM_block_jittor, num_blocks=num_blocks,
                               nb_filter=nb_filter, deep_supervision=True)
    model.apply(xavier_init_jittor)
    checkpoint = jt.load('result_jt/' + model_dir)
    model.load_state_dict(checkpoint)
    print("Model Initializing")

    jt.clean_graph()
    jt.sync_all()
    jt.gc()
    model.eval()
    tbar = tqdm(test_data)
    losses = AverageMeter()

    with jt.no_grad():
        num = 0
        for data, labels in tbar:
            loss = 0
            if deep_supervision == True:
                preds = model(data)
                for pred in preds:
                    loss += SoftIoULoss_jt(pred, labels)
                loss /= len(preds)
                pred = preds[-1]
            else:
                pred = model(data)
                loss = SoftIoULoss_jt(pred, labels)
            num += 1

            losses.update(loss.item(), pred.size(0))
            ROC.update(pred, labels)
            mIoU.update(pred, labels)
            PD_FA.update(pred, labels)

            ture_positive_rate, false_positive_rate, recall, precision = ROC.get()
            _, mean_IOU = mIoU.get()
        FA, PD = PD_FA.get(len(test_idx))
        scio.savemat(dir + '/' + 'value_result_jt' + '/' + st_model + '_PD_FA_' + str(255),
                     {'number_record1': FA, 'number_record2': PD})

        save_result_for_test(dir, st_model, epochs, mean_IOU, recall, precision)
