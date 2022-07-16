import math
import numpy as np
import cv2
from skimage.util import img_as_ubyte


class Binarizator:
    def __init__(self, img, window, id):
        self.img = img  # input image
        self.WINDOW = window
        self.id = id  # id Binarizator

        self.EPSILON = 0.0001
        # That is the value I use for the high contrast pixel value
        self.high_contrast_pixel_value = 1

        # get height and width of the image
        self.height, self.width = img.shape

        # define the intermediate image
        self.contrast_image = None  # image after 1 step, Contrast Image Construction
        self.high_contrast_img = None  # Image after 2 step, High Constrast Pixel Detection
        self.finale_image = None  # image after 3 step, threshold applicated

    '''
    Given an image the function build its contrast image by using an image
    contrast that is calculated based on f_max (local image max) and f_min (local image min)
    '''

    def contrast_image_constructor(self):
        ret = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                f_max, f_min = self.get_min_max_intensities(i, j)
                ret[i][j] = (f_max - f_min) / (f_max + f_min + self.EPSILON)
        # the output image is flot but in order to use cv2 the image must
        # be in int form
        self.contrast_image = img_as_ubyte(ret)

    '''
    return the maxmim and minimum image intensities within a local neighborhood 
    window
    '''

    def get_min_max_intensities(self, x, y):
        # random value
        f_max = 0
        f_min = 300

        step = self.WINDOW // 2
        for i in range(max(0, x - step), min(self.height, x + step + 1)):
            for j in range(max(0, y - step), min(self.width, y + step + 1)):
                if self.img[i][j] > f_max:
                    f_max = self.img[i][j]
                if self.img[i][j] < f_min:
                    f_min = self.img[i][j]
        return f_max, f_min

    '''
    the function calculate the high contrast image pixels using the 
    Otsu's global thresholding method
    '''

    def high_contrast_image(self):
        gray = self.contrast_image
        blurred = cv2.GaussianBlur(gray, (7,7), 0)

        # applying Otsu threshdolding tecnique
        ret, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.high_contrast_img = thresh1
        #The high contrast pixel value is 1, because are the pixel above the threshold
        self.high_contrast_pixel_value = 255

    '''
    Pixels are classified based on the local threshold that is 
    estimated from the detected high contrast image pixels
    '''

    def pixels_classification(self):
        ret = np.zeros((self.height, self.width))
        window, n_min = self.parameter_estimation()
        for x in range(self.height):
            for y in range(self.width):
                n_e = self.estimate_n_e(window, x, y)
                e_mean, e_std = self.estimate_avarage_intensity(window, x, y, n_e)
                if n_e >= n_min and self.img[x][y] <= (e_mean + e_std / 2):
                    ret[x][y] = 0  # I give 0 because I use white as the text color
                else:
                    ret[x][y] = 1
        self.finale_image = ret

    '''
    The funciton estimate the number of high contrast image pixels
    within the neighborhood windows
    '''

    def estimate_n_e(self, window, x, y):
        ret = 0
        step = window // 2
        for i in range(max(0, x - step), min(self.height, x + step + 1)):
            for j in range(max(0, y - step), min(self.width, y + step + 1)):
                if self.high_contrast_img[i][j] == self.high_contrast_pixel_value:  # if the pixel is an high contrast pixel
                    ret += 1
        return ret

    def estimate_avarage_intensity(self, window, x, y, n_e):
        e_mean = 0
        e_std = 0

        # calculate the e_mean
        num = 0
        step = window // 2
        for i in range(max(0, x - step), min(self.height, x + step + 1)):
            for j in range(max(0, y - step), min(self.width, y + step + 1)):
                num += self.img[i][j] * (1 - 0 if self.high_contrast_img[i][j] == self.high_contrast_pixel_value else 1)
        e_mean = num / n_e

        # calculate e_std
        num = 0
        for i in range(max(0, x - step), min(self.height, x + step + 1)):
            for j in range(max(0, y - step), min(self.width, y + step + 1)):
                num += pow((self.img[i][j] - e_mean) * (1 - 0 if self.high_contrast_img[i][j] == self.high_contrast_pixel_value else 1), 2)
        e_std = math.sqrt(num / 2)
        return e_mean, e_std

    '''
    The function estimate the size of the window and the value n_min used 
    for the classification
    return window, n_min
    '''

    def parameter_estimation(self):
        # assume the text stroke width is 4
        t_width = 3
        window = t_width  # the size of the window should be similar to the text strole width
        n_min = window + 6  # the minimum number of high contrast pixel should be around the size of the window
        return window, n_min

    # getter function
    def get_contrast_image(self):
        return self.contrast_image

    def get_high_contrast_img(self):
        return self.high_contrast_img

    def get_final_image(self):
        return self.finale_image

    def getID(self):
        return self.id
