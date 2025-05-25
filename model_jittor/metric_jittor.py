import numpy as np
from skimage import measure
import jittor as jt


class ROCMetric_jt():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, bins):  # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric_jt, self).__init__()
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins  # 当前的阈值
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg_jt(preds, labels, score_thresh)  # class_pos：预测为真的像素数
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros([self.bins + 1])
        self.pos_arr = np.zeros([self.bins + 1])
        self.fp_arr = np.zeros([self.bins + 1])
        self.neg_arr = np.zeros([self.bins + 1])
        self.class_pos = np.zeros([self.bins + 1])


class PD_FA_jt():
    # 检出率与虚警率
    def __init__(self, bins):
        super(PD_FA_jt, self).__init__()
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)  # 目标数

    def update(self, preds, labels):

        for iBin in range(self.bins + 1):
            score_thresh = iBin * (255 / self.bins)  # 本次阈值
            predits = (preds>score_thresh).int64().numpy()  #predits = np.array((preds > score_thresh).cpu()).astype('int64')
            predits = np.reshape(predits, (256, 256))
            labelss = (labels).int64().numpy()  #labelss = np.array((labels).cpu()).astype('int64')  # P
            labelss = np.reshape(labelss, (256, 256))

            image = measure.label(predits, connectivity=2)  # 8领域搜索连通域
            coord_image = measure.regionprops(image)  # 获取连通区域的各个指标
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)   #目标数
            self.image_area_total = []  # 每个检出目标的面积
            self.image_area_match = []  # 对应的预测区域的面积
            self.distance_match = []  # 对应的质心距离
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))  # 标记区域的质心
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))  # 预测的质心
                    distance = np.linalg.norm(centroid_image - centroid_label)  # 质心间的距离
                    area_image = np.array(coord_image[m].area)  # 预测的面积
                    if distance < 3:  # 第i个标记区域与第m个检测区域匹配成功
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]  # 第m个检测区域不再加入匹配
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]  # 预测区域未匹配的面积
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)

    def get(self, img_num):

        Final_FA = self.FA / ((256 * 256) * img_num)
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


class mIoU_jt():

    def __init__(self):
        super(mIoU_jt, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy_jt(preds, labels)  # correct:正确预测的像素数；labeled:标记为正的像素数
        inter, union = batch_intersection_union_jt(preds, labels)  # inter:交集的面积;union:并集的面积
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)  # 像素级的精确度
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)  # iou
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg_jt(output, target, score_thresh):
    predict = (output.sigmoid() > score_thresh).float()
    target = target.float()

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy_jt(output, target):
    target = target.float()

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"  # target和output形状不匹配则报错

    predict = (output > 0).float()  # 没有过sigmoid函数，因此output>0就相当于sigmoid(output)>0.5
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union_jt(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    target = target.float()
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.numpy(), bins=nbins, range=(mini, maxi))  # 相交的面积
    area_pred, _ = np.histogram(predict.numpy(), bins=nbins, range=(mini, maxi))  # 预测的面积
    area_lab, _ = np.histogram(target.numpy(), bins=nbins, range=(mini, maxi))  # 标记的面积
    area_union = area_pred + area_lab - area_inter  # 预测与标记的并集的面积

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
