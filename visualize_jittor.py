from model_jittor.model_DNANet_jittor import Res_CBAM_block_jittor
from model_jittor.model_DNANet_jittor import DNANet_jittor
from model_jittor.utils_jittor import *
from model_jittor.loss_jittor import *
from model_jittor.load import *
from model_jittor.metric_jittor import *
from tqdm import tqdm
import scipy.io as scio

epchos=1500
channel_size=3
backbone="resnet_18"
deep_supervision=True
root = "dataset"
dataset = "NUDT-SIRST"
split = "50_50"
jt.flags.use_cuda = 1
lr=0.05
min_lr=1e-5
model_dir="NUDT-SIRST_DNANet_24_05_2025_20_16_48_wDS\\mIoU__NUDT-SIRST_DNANet_epoch.pkl"
st_model="NUDT-SIRST_DNANet_24_05_2025_20_16_48_wDS" #"NUDT-SIRST_DNANet_23_05_2025_18_04_56_wDS"
epochs=450

if __name__ == "__main__":
    save_prefix = f"{dataset}_DNANet"

    train_idx, test_idx, test_txt = load_dataset(root, dataset, split)
    dir = root + "\\" + dataset
    testset_jt = test_dataset(dir, test_idx=test_idx, base_size=256, crop_size=256, suffix='.png')
    test_data = jt.dataset.DataLoader(testset_jt, batch_size=1, num_workers=0, drop_last=False)

    visulization_path = dir + '/' + 'visulization_result_jt' + '/' + st_model + '_visulization_result'
    visulization_fuse_path = dir + '/' + 'visulization_result_jt' + '/' + st_model + '_visulization_fuse'
    make_visulization_dir(visulization_path, visulization_fuse_path)

    nb_filter, num_blocks = load_param(channel_size, backbone)
    model = DNANet_jittor(classes=1, in_channels=3, block=Res_CBAM_block_jittor, num_blocks=num_blocks,
                               nb_filter=nb_filter, deep_supervision=True)
    model.apply(xavier_init_jittor)
    checkpoint = jt.load('result_jt/' + model_dir)
    model.load_state_dict(checkpoint)
    print("Model Initializing")

    model.eval()
    tbar = tqdm(test_data)

    with jt.no_grad():
        num = 0
        for data, labels in tbar:
            if deep_supervision == True:
                preds = model(data)
                pred = preds[-1]
            else:
                pred = model(data)
            save_Pred_GT(pred, labels, visulization_path, test_idx, num, '.png')
            num += 1
    total_visulization_generation(dir, 'TXT', test_txt, '.png', visulization_path,
                                  visulization_fuse_path)