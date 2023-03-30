import numpy as np
import cv2
import glob
import os
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from skimage import io, transform

def read_xml():
    # read xml files and extract the coodinates of each bonding boxes
    # and return a dictionary with keys correspond to image name and
    # values correspnd to coodinates


    # put the path of xml files here
    path = "C:/Users/17654/Desktop/Andy relabeled files"
    dict = {}
    for file in os.listdir(path):
        if file.endswith('.xml'):
            tree = ET.parse(os.path.join(path,file))
            root = tree.getroot()
            temp = []
            filename = os.path.splitext(file)[0]+'.jpg'
            for object in root.findall('object'):
                n = object.find('bndbox')
                xmin = int(n.find('xmin').text)
                xmax = int(n.find('xmax').text)
                ymin = int(n.find('ymin').text)
                ymax = int(n.find('ymax').text)

                temp.append((xmin, ymin, xmax, ymax))
            dict[filename] = temp
    return dict

def crop_img(coordinates, img_path, new_dir):

    # This function takes in coodinates, image path and a new directory that
    # images are saved and crop all the bonding boxes in each image according
    # to the coodinates and save it to the new_dir.


    img = Image.open(img_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for i, coords in enumerate(coordinates):
       cropped_img = img.crop(coords)
       cropped_img.save(os.path.join(new_dir, f"{i}_{coords}.jpg"))

def save_crop_img(new_dir):

    # This function first call the read_xml() funciton to get the dictionary,
    # and then save each cropped image in new_dir

    temp = read_xml()
    # put path of the image folder here
    image_dir = "C:/Users/17654/Desktop/Images to Label/Andy"
    for image_name, coords in temp.items():
        image_path = os.path.join(image_dir, image_name)
        crop_img(coords, image_path, new_dir)

#replace this with the new destination folder path you want to save
save_crop_img("C:/Users/17654/Desktop/Cropped Image")

def coords_mids():
    dict11 = read_xml()
    combined_coords = []
    for coords in dict11.values():
        for values in coords:
            combined_coords.append(values)
    average = [sum(x) / len(x) for x in zip(*combined_coords)]
    return average

average = coords_mids()
print(average)

def resize_img():
    path1 = glob.glob("C:/Users/17654/Desktop/Cropped Image/*.jpg")
    average = coords_mids()
    width = average[2] - average[0]
    height = average[3] - average[1]
    new_img_path = "C:/Users/17654/Desktop/resized img"
    for img_path in path1:
        img = cv2.imread(img_path)
        resized_img = transform.resize(img, (height, width), preserve_range=True)
        filename = os.path.basename(img_path)
        io.imsave(f'{new_img_path}/{filename}', resized_img.astype(img.dtype))

resize_img()

def get_name():

    # A function that get the name of every name of file in
    # the desination folder

    path = "C:/Users/17654/Desktop/resized img"
    file_name = os.listdir(path)

    return file_name

def read_img():

    # A function that read every image in the destination folder
    # and return a list for future data processing

    path = glob.glob("C:/Users/17654/Desktop/resized img/*.jpg")
    img_list = []
    for img in path:
        temp = cv2.imread(img)
        img_list.append(temp)

    return img_list

def laplacian_variance(path):

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
        threshold = 1178.5249339404377
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
