import numpy as np
import math
from skimage.morphology import thin, skeletonize



class Measure:
    def __init__(self, img, img_gt):
        self.img = img  # output image, must be in range [0,1]
        self.img_gt = img_gt  # ground truth image

        # check the image size
        if len(img) != len(img_gt) or len(img[0]) != len(img_gt[0]):
            raise ValueError("The images must have the same size")

        # get height and width of the image
        self.height, self.width = img.shape

        self.f_measure_ = None
        self.p_f_measure_ = None
        self.psnr_ = None
        self.drd_ = None




    '''
    The function return the f_measure values
    '''
    def f_measure(self):
        TP, TN, FP, FN = self.tp_tn_fp_fn_calculation()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_measure = (2 * recall * precision) / (recall + precision)
        self.f_measure_ = f_measure

    '''
    The funciton return the p F measure using the p value in input
    '''
    def p_f_measure(self):
        TP, TN, FP, FN = self.tp_tn_fp_fn_calculation()

        #Get the skeletonized GT
        img_sk = skeletonize(1 - self.img_gt)

        ptp = np.zeros(self.img_gt.shape)
        ptp[(self.img == 0) & (img_sk == 0)] = 1
        numptp = ptp.sum()

        precall = numptp / np.sum(1 - img_sk)
        precision = TP / (TP + FP)

        p_f_m = (2 * precall * precision) / (precall + precision)
        self.p_f_measure_ = p_f_m

    '''
    the funciton calculate the Peak Signal-to-noise ratio
    '''
    def psnr(self):
        #calculate the MSE
        tmp = np.zeros(self.img.shape)
        tmp[self.img != self.img_gt] = 1
        sum = tmp.sum()

        mse = sum / (self.height * self.width)
        # I chose C = 1
        C = 1
        psnr_ = 10 * np.log(C/mse)
        self.psnr_ = psnr_

    '''
    This method calculate the Distance-Reciprocal Distortion Measure
    '''
    def drd(self):
        # calculate the weight matrix
        n = 2
        m = 2 * n + 1
        ic = jc = ((m+1)/2) - 1
        w = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i != ic or j != jc:
                    w[i][j] = 1 / math.sqrt(math.pow((i - ic), 2) + math.pow((j - jc), 2))
        wnm = w / w.sum()

        #estimate nubn
        nubn = 0
        block_size = 8
        for i in range(0, self.height, block_size):
            for j in range(0, self.width, block_size):
                i1 = min(i + block_size - 1, self.height - 1)
                j1 = max(j + block_size - 1, self.width - 1)
                block_dim = (i1 - i+1) * (j1 - j + 1) 
                block = self.img_gt[i:i1,  j:j1]
                block_sum = np.sum(block)
                '''
                if block_sum = 0 -> means all the pixel are 0
                if block_sum = block_dim -> means all the pixel are 255
                '''
                if block_sum > 0 and block_sum < block_dim:
                    nubn+= 1


        #calculate the flipped pixel
        neg = np.zeros(self.img.shape)
        neg[self.img_gt != self.img] = 1
        y, x = np.unravel_index(np.flatnonzero(neg), self.img.shape)

        #calculate DRDk
        drd_sum = 0
        tmp = np.zeros(w.shape)
        for i in range(min(1, len(y))):
            tmp[:,:] = 0
            #calculate the coordinate of the block
            x1 = max(0, x[i] - n)
            y1 = max(0, y[i] - n)
            x2 = min(self.width - 1, x[i] + n)
            y2 = min(self.height - 1, y[i] + n)

            yy1 = y1 - y[i] + n
            yy2 = y2 - y[i] + n
            xx1 = x1 - x[i] + n
            xx2 = x2 - x[i] + n

            tmp[yy1:yy2 + 1, xx1:xx2 + 1] = np.abs(self.img[y[i], x[i]] - self.img_gt[y1: y2 +1, x1: x2 + 1])
            tmp *= wnm

            drd_sum += np.sum(tmp)
        drd = drd_sum / nubn
        self.drd_ = drd

    '''
    The function calculate:
    - True Positive
    - True Negative
    - False Positive
    - False Negative
    I consider the following values:
    - Positive = 255
    - Negative = 0
    '''
    def tp_tn_fp_fn_calculation(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        p = 0
        n = 1
        for i in range(self.height):
            for j in range(self.width):
                if self.img[i][j] == p and self.img_gt[i][j] == p:
                    TP += 1
                if self.img[i][j] == n and self.img_gt[i][j] == n:
                    TN += 1
                if self.img[i][j] == p and self.img_gt[i][j] == n:
                    FP += 1
                if self.img[i][j] == n and self.img_gt[i][j] == p:
                    FN += 1
        return TP, TN, FP, FN

    def getFmeasure(self):
        return self.f_measure_

    def getPFmeasure(self):
        return self.p_f_measure_

    def getPSNR(self):
        return self.psnr_

    def getDRD(self):
        return self.drd_


