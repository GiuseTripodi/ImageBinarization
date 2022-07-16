import os
import matplotlib
from matplotlib import image
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from binarization.binarization_functions import Binarizator
from binarization.evaluation_measure import Measure
from skimage.util import img_as_ubyte
from skimage.util import img_as_float32


'''
The following function loads all the images, for each image it performs binarization and otsu technique. At the end the results of the measurements are shown.
'''
def comparation_test():
    # load images
    path = "Data/ue1"
    photo_name = "F2s.png"
    # get all the file in the directory
    arr = os.listdir(path)
    dir = [file for file in arr if os.path.isdir(os.path.join(path, file))]

    img = {}
    gt_img = {}

    for i in range(0, len(dir) - 1):
        img[dir[i]] = matplotlib.image.imread(os.path.join(path, dir[i], photo_name))
        gt_img[dir[i]] = matplotlib.image.imread(os.path.join(path, dir[i], f"{dir[i]}GT.png"))

    #Binarization
    window = 3
    path = "object"
    binarizators = {}
    for i in img.keys():
        print(f"Image: {i}")
        if os.path.isfile(os.path.join(path, i)):
            # if the binarizator on the image already exist
            print(f"file {os.path.join(path, i)} present")
            with open(os.path.join(path, i), "rb") as binarizator_file:
                bin = pickle.load(binarizator_file)

        # the file does not exist
        else:
            print(f"file {os.path.join(path, i)} NOT present")
            bin = Binarizator(img[i], window, i)
            # do the binarization
            bin.contrast_image_constructor()
            bin.high_contrast_image()
            bin.pixels_classification()

            # load on file the binarizator
            with open(os.path.join(path, i), "wb") as file:
                pickle.dump(bin, file)
        binarizators[i] = bin

    #OTSU Binarization
    output_otsu = {}
    for i in img.keys():
        image = img_as_ubyte(img[i])
        ret, thresh = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        output_otsu[i] = thresh


    #Measures
    # f_measure
    f_measure = 0
    f_measure_otsu = 0

    # p_f_measure
    pf_measure = 0
    pf_measure_otsu = 0

    # PSNR
    psnr = 0
    psnr_otsu = 0

    # DRD
    drd = 0
    drd_otsu = 0

    path = "measures"
    p = 2
    # del binarizators["z582"]
    for i in binarizators.keys():
        print(f"image: {i}")
        if os.path.isfile(os.path.join(path, i)):
            # if the binarizator on the image already exist
            print(f"file {os.path.join(path, i)} present")
            with open(os.path.join(path, i), "rb") as file:
                m = pickle.load(file)

        # the file does not exist
        else:
            print(f"file {os.path.join(path, i)} NOT present")
            m = Measure(binarizators[i].get_final_image(), gt_img[i])
            m.f_measure()
            m.p_f_measure(p)
            m.psnr()
            m.drd()

            # load on file
            with open(os.path.join(path, i), "wb") as file:
                pickle.dump(m, file)

        f_measure += m.getFmeasure()
        print(m.getFmeasure())
        pf_measure += m.getPFmeasure()
        psnr += m.getPSNR()
        drd += m.getDRD()

    f_measure /= len(binarizators)
    pf_measure /= len(binarizators)
    psnr /= len(binarizators)
    drd /= len(binarizators)

    for i in output_otsu.keys():
        # otsu
        print(f"image: {i}")
        file_name = i + "_otsu"
        if os.path.isfile(os.path.join(path, file_name)):
            # if the binarizator on the image already exist
            print(f"file {os.path.join(path, file_name)} present")
            with open(os.path.join(path, file_name), "rb") as file:
                m_otsu = pickle.load(file)

        # the file does not exist
        else:
            print(f"file {os.path.join(path, file_name)} NOT present")
            m_otsu = Measure(output_otsu[i], gt_img[i])
            m_otsu.f_measure()
            m_otsu.p_f_measure(p)
            m_otsu.psnr()
            m_otsu.drd()

            # load on file
            with open(os.path.join(path, file_name), "wb") as file:
                pickle.dump(m_otsu, file)

        f_measure_otsu += m_otsu.getFmeasure()
        pf_measure_otsu += m_otsu.getPFmeasure()
        psnr_otsu += m_otsu.getPSNR()
        drd_otsu += m_otsu.getDRD()

    f_measure_otsu /= len(output_otsu)
    pf_measure_otsu /= len(output_otsu)
    psnr_otsu /= len(output_otsu)
    drd_otsu /= len(output_otsu)

    #Results
    # print measures
    print("\n")
    print(f"f measure: {f_measure}")
    print(f"Pf measure: {pf_measure}")
    print(f"PSNR: {psnr}")
    print(f"DRD: {drd} \n")

    print(f"Otsu measure")
    print(f"f measure: {f_measure_otsu}")
    print(f"Pf measure: {pf_measure_otsu}")
    print(f"PSNR: {psnr_otsu}")
    print(f"DRD: {drd_otsu} \n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    comparation_test()

