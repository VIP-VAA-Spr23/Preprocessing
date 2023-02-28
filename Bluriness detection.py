import numpy as np
import cv2
import glob
import os
import pandas as pd

def get_name():

    # A function that get the name of every name of file in
    # the desination folder

    path = "copy the path here"
    file_name = os.listdir(path)

    return file_name

def read_img():

    # A function that read every image in the destination folder
    # and return a list for future data processing

    path = glob.glob("copy the path here*.jpg")
    img_list = []
    for img in path:
        temp = cv2.imread(img)
        img_list.append(temp)

    return img_list

def laplacian_variance(img):

    # A function that calculate the laplacian variance of img
    # this is the main tool used for analysis bluriness of img

    lp_var = cv2.Laplacian(img, cv2.CV_64F).var()

    return lp_var

if __name__ == "__main__":

    # main function that display result
    # threshold set according to the median of variance of images

    i_list = read_img()
    blur_list = []
    mid = []
    for img in i_list:
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cmp = laplacian_variance(cvt)
        blur_list.append(cmp)
        threshold = 1954.1616589632376
        if cmp > threshold:
            dis = "Above mid"
        else:
            dis = "Below mid"
        mid.append(dis)
    median = np.median(blur_list)
    name = get_name()

    # set up dictionary which has keys with image name and values
    # are blurriness of image and whether or not it below or above median
    dictionary = dict(zip(name, zip(blur_list, mid)))
    print(dictionary)

    # set up excel to display result
    df = pd.DataFrame(list(dictionary.items()), columns=['img_name','blur value'])
    df.to_excel('blur detection.xlsx', index=False)