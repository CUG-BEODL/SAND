#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author ：hhx
@Date ：2022/9/8 14:59 
@Description ：评价指标
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Evaluator(object):
    """语义分割类指标"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):  # OA
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):  # 召回率
        Acc_classes = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc_classes)
        return Acc_classes, Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def F1(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        f1 = 2 * precision * recall / (precision + recall)
        f1 = np.nanmean(f1)
        return f1

    def Kappa(self):
        p_o = self.Pixel_Accuracy()
        pre = np.sum(self.confusion_matrix, axis=0)
        label = np.sum(self.confusion_matrix, axis=1)
        p_e = (pre * label).sum() / (self.confusion_matrix.sum() * self.confusion_matrix.sum())
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # print(gt_image, pre_image)
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        # print(count)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Score(object):
    def __init__(self, classfication):
        self.classfication = classfication
        self.classfication.reset()
        self.all_change_sample_num = 0
        self.correct_sample_num = 0

    def add(self, label_seq, predicted_seq):
        if label_seq.shape[0] != 1:
            self.all_change_sample_num += 1
            if np.array_equal(predicted_seq, label_seq):
                self.correct_sample_num += 1

        if label_seq.shape[0] != 1:
            label_seq = np.array([6])
        if predicted_seq.shape[0] != 1:
            predicted_seq = np.array([6])

        self.classfication.add_batch(label_seq, predicted_seq)

    def getScore(self):
        score = {
            "OA": round(self.classfication.Pixel_Accuracy() * 100, 2),
            "AA": round(self.classfication.Pixel_Accuracy_Class()[1] * 100, 2),
            "F1": round(self.classfication.F1() * 100, 2),
            "Kappa": round(self.classfication.Kappa() * 100, 2),
            "CT": round(self.correct_sample_num * 100 / self.all_change_sample_num, 2)
        }
        return score
