import jittor as jt
from jittor import transform
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from datetime import datetime
import os
import shutil
from  matplotlib import pyplot as plt

def xavier_init_jittor(module):
    #if isinstance(module,jt.nn.Conv2d):
        #jt.init.xavier_gauss_(module.weight)
    classname = module.__class__.__name__
    if classname.find('Conv2d') != -1:
        #jt.init.xavier_gauss_(module.weight.data)
        module.weight.xavier_gauss_()

class train_dataset(jt.dataset.Dataset):
    def __init__(self,dir,train_idx,base_size=512,crop_size=256,suffix='.png'):
        super(train_dataset,self).__init__()
        self.img_id=train_idx
        self.mask_dir=dir+"\\masks"
        self.img_dir=dir+"\\images"
        self.base_size=base_size
        self.crop_size=crop_size
        self.suffix=suffix
        self.total_len=len(train_idx)

    def _sync_transform(self, img, mask):
        # 数据增强
        # random mirror（以一定概率翻转）
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)（缩放）
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop（不足crop_size的补齐到crop_size）
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size（随机位置裁剪到crop_size）
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP（一定概率高斯模糊）
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = img, np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):

        img_id = self.img_id[idx]
        img_path = self.img_dir + '/' + img_id + self.suffix
        label_path = self.mask_dir + '/' + img_id + self.suffix

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        # synchronized transform
        random.seed(idx)
        #print(f"jittor: {idx}")
        img, mask = self._sync_transform(img, mask)

        #image preprocess
        norm=transform.ImageNormalize(mean=[.485, .456, .406],std=[.229, .224, .225])
        img=norm(img)

        #mask to jt
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0  # mask变换成1，0变量

        return img, jt.array(mask)

class test_dataset(jt.dataset.Dataset):
    def __init__(self,dir,test_idx,base_size=512,crop_size=256,suffix='.png'):
        super(test_dataset,self).__init__()
        self.img_id = test_idx
        self.mask_dir = dir + "\\masks"
        self.img_dir = dir + "\\images"
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.total_len = len(test_idx)

    def _sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        img, mask = img, np.array(mask,dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id = self.img_id[idx]
        img_path = self.img_dir + '/' + img_id + self.suffix
        label_path = self.mask_dir + '/' + img_id + self.suffix

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask)

        # image preprocess
        norm = transform.ImageNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        img = norm(img)

        # mask to jt
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0  # mask变换成1，0变量

        return img, jt.array(mask)

def makdir(dataset,deep_supervision):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = f"{dataset}_DNANet_{dt_string}_wDS"
    else:
        save_dir = f"{dataset}_DNANet_{dt_string}_woDS"
    os.makedirs(f"result_jt\\{save_dir}", exist_ok=True)
    return save_dir

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    jt.save(state, os.path.join(save_path,filename))

def save_train_log(save_dir,epchos,channel_size,backbone,deep_supervision,dataset,split,lr,min_lr):
    with open(f"result_jt\\{save_dir}\\train_log.txt" ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        f.write(f"epchos:--{epchos}\n")
        f.write(f"channel_size:--{channel_size}\n")
        f.write(f"backbone:--{backbone}\n")
        f.write(f"deep_supervision:--{deep_supervision}\n")
        f.write(f"dataset:--{dataset}\n")
        f.write(f"split:--{split}\n")
        f.write(f"lr:--{lr}\n")
        f.write(f"min_lr:--{min_lr}")
    return

def save_test_result(dt_string, epoch,train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model(mean_IOU, best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
    if mean_IOU > best_iou:
        save_mIoU_dir = 'result_jt/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result_jt/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        save_test_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt(net, save_path='result_jt/' + save_dir,filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pkl')

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'value_result_jt'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' +'value_result_jt'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    #create fused image
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.Resampling.LANCZOS) #img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')



def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    #predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = (pred > 0).numpy().astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    #labelsss = np.uint8(labelsss.cpu())
    labelsss=np.uint8(labelsss.numpy())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    #predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = (pred > 0).numpy().astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)

    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()


class Adagrad(jt.optim.Optimizer):
    """ Adagrad Optimizer.
    Args:
        params(list): parameters of model.
        lr(float): learning rate.
        eps(float): term added to the denominator to avoid division by zero, default 1e-8.

    Example:
        optimizer = nn.Adagrad(model.parameters(), lr)
        optimizer.step(loss)
    """

    def __init__(self, params, lr=1e-2, eps=1e-10):
        super().__init__(params, lr)
        self.eps = eps

        # 初始化梯度平方累积量
        for pg in self.param_groups:
            values = pg["sum_squares"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["sum_squares"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            for p, g, s in zip(pg["params"], pg["grads"], pg["sum_squares"]):
                if p.is_stop_grad(): continue
                # 累积梯度平方
                s.update(s + g * g)
                # 更新参数
                p.update(p - lr * g / (jt.sqrt(s) + eps))
        self.post_step()