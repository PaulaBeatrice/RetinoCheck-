import csv
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multiprocessing.pool import ThreadPool
from tensorflow.python.keras.utils.np_utils import to_categorical


def data_processing():
    pd.set_option('display.max_colwidth', None)
    os.listdir('D:\FACULTATE\FACULTATE\LICENTA\proiect\data')

    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_1.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images')
    train_data['file_name'] = train_data['id_code'].apply(lambda x: x + ".png")

    train_data['file_path'] = train_data['id_code'].apply(lambda x: os.path.join(train_imgs, x + ".png"))
    train_data['diagnosis'] = to_categorical(train_data['diagnosis'], num_classes=5)

    test_data = pd.read_csv(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test.csv')
    test_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images')
    test_data['file_name'] = test_data['id_code'].apply(lambda x: x + ".png")
    test_data['file_path'] = test_data['id_code'].apply(lambda x: os.path.join(test_imgs, x + ".png"))
    test_data['diagnosis'] = to_categorical(test_data['diagnosis'], num_classes=5)

    train_data['diagnosis'] = train_data['diagnosis'].astype(str)
    test_data['diagnosis'] = test_data['diagnosis'].astype(str)

    return train_data, test_data


def data_processing_aug():
    pd.set_option('display.max_colwidth', None)
    os.listdir('D:\FACULTATE\FACULTATE\LICENTA\proiect\data')

    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_2.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_aug')

    train_data['file_name'] = train_data['id_code'].apply(lambda x: x + ".png")

    train_data['file_path'] = train_data['id_code'].apply(lambda x: os.path.join(train_imgs, x + ".png"))
    train_data['diagnosis'] = to_categorical(train_data['diagnosis'], num_classes=5)

    test_data = pd.read_csv(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test.csv')
    test_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images')
    test_data['file_name'] = test_data['id_code'].apply(lambda x: x + ".png")
    test_data['file_path'] = test_data['id_code'].apply(lambda x: os.path.join(test_imgs, x + ".png"))
    test_data['diagnosis'] = to_categorical(test_data['diagnosis'], num_classes=5)

    train_data['diagnosis'] = train_data['diagnosis'].astype(str)
    test_data['diagnosis'] = test_data['diagnosis'].astype(str)

    return train_data, test_data


def data_processing_aug_clahe():
    pd.set_option('display.max_colwidth', None)
    os.listdir('D:\FACULTATE\FACULTATE\LICENTA\proiect\data')

    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_2.csv')
    train_imgs = os.path.abspath(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe')

    train_data['file_name'] = train_data['id_code'].apply(lambda x: x + ".png")

    train_data['file_path'] = train_data['id_code'].apply(lambda x: os.path.join(train_imgs, x + ".png"))
    train_data['diagnosis'] = to_categorical(train_data['diagnosis'], num_classes=5)

    test_data = pd.read_csv(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test.csv')
    test_imgs = os.path.abspath(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images_apply_clahe')
    test_data['file_name'] = test_data['id_code'].apply(lambda x: x + ".png")
    test_data['file_path'] = test_data['id_code'].apply(lambda x: os.path.join(test_imgs, x + ".png"))
    test_data['diagnosis'] = to_categorical(test_data['diagnosis'], num_classes=5)

    train_data['diagnosis'] = train_data['diagnosis'].astype(str)
    test_data['diagnosis'] = test_data['diagnosis'].astype(str)

    return train_data, test_data


def data_processing_aug_ddr():
    pd.set_option('display.max_colwidth', None)
    os.listdir('D:\FACULTATE\FACULTATE\LICENTA\proiect\data')

    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_aug.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_AUG')

    train_data['file_name'] = train_data['id_code'].apply(lambda x: x + ".jpg")

    train_data['file_path'] = train_data['id_code'].apply(lambda x: os.path.join(train_imgs, x + ".jpg"))
    train_data['diagnosis'] = to_categorical(train_data['diagnosis'], num_classes=5)

    test_data = pd.read_csv(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test.csv')
    test_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images')
    test_data['file_name'] = test_data['id_code'].apply(lambda x: x + ".png")
    test_data['file_path'] = test_data['id_code'].apply(lambda x: os.path.join(test_imgs, x + ".png"))
    test_data['diagnosis'] = to_categorical(test_data['diagnosis'], num_classes=5)

    train_data['diagnosis'] = train_data['diagnosis'].astype(str)
    test_data['diagnosis'] = test_data['diagnosis'].astype(str)

    return train_data, test_data


def data_processing_aug_ddr_aptos():
    pd.set_option('display.max_colwidth', None)
    os.listdir('D:\FACULTATE\FACULTATE\LICENTA\proiect\data')

    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_aug.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_images_aug')

    train_data['file_name'] = train_data['id_code'].apply(lambda x: x + ".png")

    train_data['file_path'] = train_data['id_code'].apply(lambda x: os.path.join(train_imgs, x + ".jpg"))
    train_data['diagnosis'] = to_categorical(train_data['diagnosis'], num_classes=5)

    test_data = pd.read_csv(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\test.csv')
    test_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\test_images')
    test_data['file_name'] = test_data['id_code'].apply(lambda x: x + ".png")
    test_data['file_path'] = test_data['id_code'].apply(lambda x: os.path.join(test_imgs, x + ".png"))
    test_data['diagnosis'] = to_categorical(test_data['diagnosis'], num_classes=5)

    train_data['diagnosis'] = train_data['diagnosis'].astype(str)
    test_data['diagnosis'] = test_data['diagnosis'].astype(str)

    return train_data, test_data


def resize_img(file):
    input_path = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images',
                              '{}.png'.format(file))
    output_path = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_resized',
                               '{}.png'.format(file))
    img = cv2.imread(input_path)
    cv2.imwrite(output_path, cv2.resize(img, (512, 512)))


def resize_all_imgs(nr, imgs):
    """
    Salveaza toate imaginile in folderul nou
    :param nr: nr de procese ce ruleaza in paralel
    :param imgs: lista de img
    :return:
    """
    print(f'MESSAGE: Running {nr} img')
    results = ThreadPool(nr).map(resize_img, imgs)
    return results


def graphics_distribution(data, title):
    data_group = pd.DataFrame(data.groupby('diagnosis').agg('size').reset_index())
    data_group.columns = ['diagnosis', 'count']

    sns.set(rc={'figure.figsize': (10, 5)}, style="whitegrid")

    colors = sns.color_palette("rocket", len(data_group['diagnosis']))

    sns.barplot(x='diagnosis', y='count', data=data_group, palette=colors, hue="diagnosis")
    plt.title('Class Distribution ' + title)
    plt.show()


def conv_gray(img):
    """
    converteste o imagine color la gray scale image
    :param img:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    return img


def view_imgs(data, pts_per_class, color_scale):
    data = data.groupby('diagnosis', group_keys=False).apply(lambda data: data.sample(pts_per_class))
    data = data.reset_index(drop=True)

    plt.rcParams["axes.grid"] = False
    for pt in range(pts_per_class):
        f, axarr = plt.subplots(1, 5, figsize=(15, 15))
        axarr[0].set_ylabel("Sample Data Points")

        temp = data[data.index.isin(
            [pt + (pts_per_class * 0), pt + (pts_per_class * 1), pt + (pts_per_class * 2), pt + (pts_per_class * 3),
             pt + (pts_per_class * 4)])]
        for i in range(5):
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(temp.file_path.iloc[i]))
                axarr[i].imshow(img, cmap=color_scale)
            else:
                axarr[i].imshow(Image.open(temp.file_path.iloc[i]).resize((200, 200)))
            axarr[i].set_xlabel('Class ' + str(temp.diagnosis.iloc[i]))

        plt.show()


def enhancing_clahe(img):
    """
    imbunatatim imaginile cu contrast scazut
    :return:
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # aplicam clahe pe fiecare canal de culoare
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    ench_channels = [clahe.apply(channel) for channel in cv2.split(img)]

    ench_img = cv2.merge(ench_channels)
    return ench_img


def view_imgs_enhanced(data, pts_per_class, color_scale):
    data = data.groupby('diagnosis', group_keys=False).apply(lambda data: data.sample(pts_per_class))
    data = data.reset_index(drop=True)

    plt.rcParams["axes.grid"] = False
    for pt in range(pts_per_class):
        f, axarr = plt.subplots(2, 5, figsize=(30, 15))
        axarr[0, 0].set_ylabel("Original")
        axarr[1, 0].set_ylabel("Enhanced")

        temp = data[data.index.isin(
            [pt + (pts_per_class * 0), pt + (pts_per_class * 1), pt + (pts_per_class * 2), pt + (pts_per_class * 3),
             pt + (pts_per_class * 4)])]
        for i in range(5):
            img_path = temp.file_path.iloc[i]

            # afisare img initiala
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                axarr[0, i].imshow(img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                axarr[0, i].imshow(img)
            axarr[0, i].set_xlabel('Original Class ' + str(temp.diagnosis.iloc[i]))

            # afisare img enchancing
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                ench_img = enhancing_clahe(img)
                axarr[1, i].imshow(ench_img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                ench_img = enhancing_clahe(img)
                axarr[1, i].imshow(ench_img)
            axarr[1, i].set_xlabel('Enhanced Class ' + str(temp.diagnosis.iloc[i]))

        plt.show()


def apply_Gaussian_filter(img):
    """
    aplicam filtrul gaussian pentru noise removing
    :param img:
    :return:
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    gauss_img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128)
    return gauss_img


def view_imgs_enhanced_gaussian(data, pts_per_class, color_scale):
    data = data.groupby('diagnosis', group_keys=False).apply(lambda data: data.sample(pts_per_class))
    data = data.reset_index(drop=True)

    plt.rcParams["axes.grid"] = False
    for pt in range(pts_per_class):
        f, axarr = plt.subplots(4, 5, figsize=(20, 15))
        axarr[0, 0].set_ylabel("Original")
        axarr[1, 0].set_ylabel("Enhanced")
        axarr[1, 0].set_ylabel("Gaussian filter")
        axarr[2, 0].set_ylabel("Gaussian-Enhanced filter")

        temp = data[data.index.isin(
            [pt + (pts_per_class * 0), pt + (pts_per_class * 1), pt + (pts_per_class * 2), pt + (pts_per_class * 3),
             pt + (pts_per_class * 4)])]
        for i in range(5):
            img_path = temp.file_path.iloc[i]

            # afisare img initiala
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                axarr[0, i].imshow(img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                axarr[0, i].imshow(img)
            axarr[0, i].set_xlabel('Original Class ' + str(temp.diagnosis.iloc[i]))

            # afisare img enchancing
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                ench_img = enhancing_clahe(img)
                gauss_enh_img = apply_Gaussian_filter(ench_img)
                gauss_img = apply_Gaussian_filter(img)
                axarr[1, i].imshow(ench_img, cmap=color_scale)
                axarr[2, i].imshow(gauss_img, cmap=color_scale)
                axarr[3, i].imshow(gauss_enh_img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                ench_img = enhancing_clahe(img)
                gauss_enh_img = apply_Gaussian_filter(ench_img)
                gauss_img = apply_Gaussian_filter(img)
                axarr[1, i].imshow(ench_img)
                axarr[2, i].imshow(gauss_img)
                axarr[3, i].imshow(gauss_enh_img)
            axarr[1, i].set_xlabel('Enhanced Class ' + str(temp.diagnosis.iloc[i]))
            axarr[2, i].set_xlabel('Gaussian Filter Class ' + str(temp.diagnosis.iloc[i]))
            axarr[3, i].set_xlabel('Gaussian and Enhanced Filter Class ' + str(temp.diagnosis.iloc[i]))

        plt.show()


def crop_Image(img, tol=7):
    """
    Elimina zonele negre din imagine
    :param tol:
    :param img:
    :return:
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:
            return img  # returnam imaginea originala, pentru ca e prea intunecata
        else:
            img1 = img[:, :, 0][
                np.ix_(mask.any(1), mask.any(0))]  # subimagini pentru fiecare dintre cele 3 canale de culoare
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def apply_Gaussian_filter_Circle_cropping(img, sigmaX):
    """
    Se da path ul imaginii
    :param img:
    :param sigmaX:
    :return:
    """
    img = cv2.imread(img)
    img = crop_Image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


def apply_Gaussian_filter_Circle_cropping_v2(img, sigmaX):
    """
    Se da imaginea
    :param img:
    :param sigmaX:
    :return:
    """
    img = crop_Image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


def view_imgs_gauss_crop(data, pts_per_class, color_scale):
    data = data.groupby('diagnosis', group_keys=False).apply(lambda data: data.sample(pts_per_class))
    data = data.reset_index(drop=True)

    plt.rcParams["axes.grid"] = False
    for pt in range(pts_per_class):
        f, axarr = plt.subplots(3, 5, figsize=(20, 15))
        axarr[0, 0].set_ylabel("Original")
        axarr[1, 0].set_ylabel("Gaussian filter")
        axarr[2, 0].set_ylabel("Gaussian-Cropped filter")
        sigmaX = 30

        temp = data[data.index.isin(
            [pt + (pts_per_class * 0), pt + (pts_per_class * 1), pt + (pts_per_class * 2), pt + (pts_per_class * 3),
             pt + (pts_per_class * 4)])]
        for i in range(5):
            img_path = temp.file_path.iloc[i]

            # afisare img initiala
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                axarr[0, i].imshow(img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                axarr[0, i].imshow(img)
            axarr[0, i].set_xlabel('Original Class ' + str(temp.diagnosis.iloc[i]))

            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                gauss_img = apply_Gaussian_filter(img)
                gauss_crop_img = apply_Gaussian_filter_Circle_cropping(img_path, sigmaX)
                axarr[1, i].imshow(gauss_img, cmap=color_scale)
                axarr[2, i].imshow(gauss_crop_img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                gauss_img = apply_Gaussian_filter(img)
                gauss_crop_img = apply_Gaussian_filter_Circle_cropping(img_path, sigmaX)
                axarr[1, i].imshow(gauss_img)
                axarr[2, i].imshow(gauss_crop_img, cmap=color_scale)
            axarr[1, i].set_xlabel('Gaussian Filter Class ' + str(temp.diagnosis.iloc[i]))
            axarr[2, i].set_xlabel('Gaussian and Cropped Filter Class ' + str(temp.diagnosis.iloc[i]))

        plt.show()


def normalise_img(img):
    img = cv2.imread(img, 0)
    normalised_img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalised_img


def view_imgs_normalised(data, pts_per_class, color_scale):
    data = data.groupby('diagnosis', group_keys=False).apply(lambda data: data.sample(pts_per_class))
    data = data.reset_index(drop=True)

    plt.rcParams["axes.grid"] = False
    for pt in range(pts_per_class):
        f, axarr = plt.subplots(2, 5, figsize=(20, 15))
        axarr[0, 0].set_ylabel("Original")
        axarr[1, 0].set_ylabel("Normalised")

        temp = data[data.index.isin(
            [pt + (pts_per_class * 0), pt + (pts_per_class * 1), pt + (pts_per_class * 2), pt + (pts_per_class * 3),
             pt + (pts_per_class * 4)])]
        for i in range(5):
            img_path = temp.file_path.iloc[i]

            # afisare img initiala
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(img_path))
                axarr[0, i].imshow(img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                axarr[0, i].imshow(img)
            axarr[0, i].set_xlabel('Original Class ' + str(temp.diagnosis.iloc[i]))

            if color_scale == 'gray':
                normalised_img = normalise_img(img_path)
                axarr[1, i].imshow(normalised_img, cmap=color_scale)
            else:
                img = Image.open(img_path).resize((200, 200))
                normalised_img = normalise_img(img_path)
                axarr[1, i].imshow(normalised_img)
            axarr[1, i].set_xlabel('Normalised' + str(temp.diagnosis.iloc[i]))

        plt.show()


def augmentations(img, nr):
    """
    Generam nr augmentari pt imaginea img
    :param img:
    :param nr:
    :return:
    """
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                                 horizontal_flip=True)

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))

    plt.figure(figsize=(10, 10))

    plt.subplot(1, nr + 1, 1)
    plt.imshow(img)
    plt.title('Original')

    img_arr = img.reshape((1,) + img.shape)

    i = 0
    for img_iterator in datagen.flow(x=img_arr, batch_size=1):
        i = i + 1
        if i > nr:
            break
        plt.subplot(1, nr + 1, i + 1)
        augmented_img = img_iterator.reshape(img_arr[0].shape).astype(np.uint8)
        plt.imshow(augmented_img)
        plt.title('Augmented ' + str(i))

    plt.show()


def view_imgs_augmentations(data, pts_per_class):
    data = data.groupby('diagnosis', group_keys=False).apply(lambda data: data.sample(pts_per_class))
    data = data.reset_index(drop=True)

    for pt in range(pts_per_class):
        temp = data[data.index.isin(
            [pt + (pts_per_class * 0), pt + (pts_per_class * 1), pt + (pts_per_class * 2), pt + (pts_per_class * 3),
             pt + (pts_per_class * 4)])]
        for i in range(5):
            img_path = temp.file_path.iloc[i]
            augmentations(img_path, 4)


def process_imgs(file):
    """
    Aplica filtrul gausian si circle crop pe imaginile dintr-un folder
    :param file:
    :return: salveaza imaginile intr-un nou folder
    """
    input_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images',
                              '{}.png'.format(file))
    output_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_filtered',
                               '{}.png'.format(file))
    img = apply_Gaussian_filter_Circle_cropping(input_file, 30)

    cv2.imwrite(output_file, cv2.resize(img, (512, 512)))


def process_imgs_test(file):
    """
    Aplica filtrul gausian si circle crop pe imaginile dintr-un folder
    :param file:
    :return: salveaza imaginile intr-un nou folder
    """
    input_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images',
                              '{}.png'.format(file))
    output_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images_filtered',
                               '{}.png'.format(file))
    img = apply_Gaussian_filter_Circle_cropping(input_file, 30)

    cv2.imwrite(output_file, cv2.resize(img, (512, 512)))


def multi_process_ims(nr, imgs):
    results = ThreadPool(nr).map(process_imgs, imgs)
    return results


def multi_process_ims_test(nr, imgs):
    results = ThreadPool(nr).map(process_imgs_test, imgs)
    return results


def process_imgs_v2(file):
    """
    Aplica enhanced, filtrul gausian si circle crop pe imaginile dintr-un folder
    :param file:
    :return: salveaza imaginile intr-un nou folder
    """
    input_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images',
                              '{}.png'.format(file))
    output_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images'
                               r'\train_images_filtered_v2', '{}.png'.format(file))
    img = cv2.imread(input_file)
    img = enhancing_clahe(img)
    img = apply_Gaussian_filter_Circle_cropping_v2(img, 30)

    cv2.imwrite(output_file, cv2.resize(img, (512, 512)))


def multi_process_ims_v2(nr, imgs):
    results = ThreadPool(nr).map(process_imgs_v2, imgs)
    return results


def rotate_image(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    return rotated_img


def flip_image(img, flip_code):
    flipped_img = cv2.flip(img, flip_code)
    return flipped_img


def shear_image(img, shear_range):
    rows, cols = img.shape[:2]
    shear_factor = np.random.uniform(-shear_range, shear_range)
    M = np.array([[1, shear_factor, 0], [0, 1, 0]])
    sheared_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return sheared_img


def translate_image(img, tx, ty):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    return translated_img


def zoom_image(image, zoom_factor):
    height, width = image.shape[:2]
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image


def change_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] + value
    hsv[hsv[:, :, 2] > 255] = 255
    hsv[hsv[:, :, 2] < 0] = 0
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def change_contrast(image, contrast_factor):
    factor = (contrast_factor - 1) * 0.5
    inv_gamma = 1.0 / (1.0 + factor)
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def color_balance(image, factors):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0]
    hsv[:, :, 1] = hsv[:, :, 1] * factors[1]
    hsv[:, :, 2] = hsv[:, :, 2] * factors[2]
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def change_saturation(image, saturation_factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def gaussian_blur(image, ksize):
    return cv2.GaussianBlur(image, ksize, 0)


def augment_train_data():
    """
    pt diagnostic 0 - fara augmentare
    pt diagnostic 1 - augmentare (inca 4) : rotation, flipping, shearing, translation
    pt diagnostic 2 - augmentare (inca 1) : rotatie
    pt diagnostic 3 - augmentare (inca 10) : rotation, flipping, shearing, translation
    pt diagnostic 4 - augmentare (inca 7): rotation, flipping, shearing, translation
    :return:
    """
    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_1.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images')

    new_train_data_dir = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images'
                                         r'\train_images_aug')
    new_train_data_csv = 'D:\\FACULTATE\\FACULTATE\\LICENTA\\proiect\\data\\APTOS\\train_2.csv'

    for index, row in train_data.iterrows():  # parcurg fisierul csv
        image_name = row['id_code']
        diagnosis = row['diagnosis']

        img_path = os.path.join(train_imgs, image_name + '.png')
        img = cv2.imread(img_path)  # imaginea corespunzatoarea instantei csv

        # adaugam imaginea originala in fisier si inregistrarea in csv
        cv2.imwrite(os.path.join(new_train_data_dir, image_name + '.png'), img)
        with open(new_train_data_csv, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(
                [image_name, diagnosis])

        if diagnosis == 0:
            # Diagnosticul imaginii e 0 => fara augmentare
            pass

        elif diagnosis == 1:
            # Diagnosticul imaginii e 1 => 4 augmentari
            for i in range(1, 5):
                augmented_img_name = f'{image_name}_aug_{i}.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 2:
            # Diagnosticul e 2 => o augmentare
            # Apply rotation
            augmented_img_name = f'{image_name}_aug_1.png'
            augmented_img_name_csv = augmented_img_name.split('.')[0]
            augmented_img = img.copy()
            augmented_img = rotate_image(augmented_img, 30)
            # Save augmented image
            cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
            # Update CSV with augmented image record
            with open(new_train_data_csv, mode='a', newline='') as results_file:
                results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow(
                    [augmented_img_name_csv, diagnosis])
        elif diagnosis == 3:
            # Diagnosticul e 3 => 10 augmentari
            for i in range(1, 11):
                augmented_img_name = f'{image_name}_aug_{i}.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                elif i == 6:
                    # Apply brightness
                    augmented_img = change_brightness(augmented_img, 30)
                elif i == 7:
                    # Apply contrast
                    augmented_img = change_contrast(augmented_img, 1.2)
                elif i == 8:
                    # Apply color balance
                    augmented_img = color_balance(augmented_img, (0.8, 1.2, 1.2))
                elif i == 9:
                    # Apply blur
                    augmented_img = gaussian_blur(augmented_img, (15, 15))
                elif i == 10:
                    # Apply saturation
                    augmented_img = change_saturation(augmented_img, 1.5)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 4:
            # Diagnosticul e 4 => 7 augmentari
            for i in range(1, 8):
                augmented_img_name = f'{image_name}_aug_{i}.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                elif i == 6:
                    # Apply brightness
                    augmented_img = change_brightness(augmented_img, 30)
                elif i == 7:
                    # Apply contrast
                    augmented_img = change_contrast(augmented_img, 1.2)
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])


def augment_train_data_ddr_dataset():
    """
    pt diagnostic 0 - fara augmentare
    pt diagnostic 1 - augmentare (inca 10) : rotation, flipping, shearing, translation
    pt diagnostic 2 - augmentare (inca 1) : rotatie
    pt diagnostic 3 - augmentare (inca 30) : rotation, flipping, shearing, translation
    pt diagnostic 4 - augmentare (inca 7): rotation, flipping, shearing, translation
    :return:
    """
    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_grading.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_grading\DR_grading')

    new_train_data_dir = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_AUG')
    new_train_data_csv = 'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_aug.csv'

    for index, row in train_data.iterrows():  # parcurg fisierul csv
        image_name = row['id_code'].split('.')[0]
        diagnosis = row['diagnosis']

        img_path = os.path.join(train_imgs, image_name + '.jpg')
        img = cv2.imread(img_path)  # imaginea corespunzatoarea instantei csv

        # adaugam imaginea originala in fisier si inregistrarea in csv
        cv2.imwrite(os.path.join(new_train_data_dir, image_name + '.jpg'), img)
        with open(new_train_data_csv, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(
                [image_name, diagnosis])

        if diagnosis == 0:
            # Diagnosticul imaginii e 0 => fara augmentare
            pass

        elif diagnosis == 1:
            # Diagnosticul imaginii e 1 => 10 augmentari
            for i in range(1, 11):
                augmented_img_name = f'{image_name}_aug_{i}.jpg'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                elif i == 6:
                    # Apply brightness
                    augmented_img = change_brightness(augmented_img, 30)
                elif i == 7:
                    # Apply contrast
                    augmented_img = change_contrast(augmented_img, 1.2)
                elif i == 8:
                    # Apply color balance
                    augmented_img = color_balance(augmented_img, (0.8, 1.2, 1.2))
                elif i == 9:
                    # Apply blur
                    augmented_img = gaussian_blur(augmented_img, (15, 15))
                elif i == 10:
                    # Apply saturation
                    augmented_img = change_saturation(augmented_img, 1.5)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 2:
            # Diagnosticul e 2 => o augmentare
            # Apply rotation
            augmented_img_name = f'{image_name}_aug_1.jpg'
            augmented_img_name_csv = augmented_img_name.split('.')[0]
            augmented_img = img.copy()
            augmented_img = rotate_image(augmented_img, 30)
            # Save augmented image
            cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
            # Update CSV with augmented image record
            with open(new_train_data_csv, mode='a', newline='') as results_file:
                results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow(
                    [augmented_img_name_csv, diagnosis])
        elif diagnosis == 3:
            # Diagnosticul e 3 => 30 augmentari ; 28
            for i in range(1, 29):
                augmented_img_name = f'{image_name}_aug_{i}.jpg'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation - 10 rotatii
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping - 3
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                elif i == 6:
                    # Apply brightness - 8
                    augmented_img = change_brightness(augmented_img, 30)
                elif i == 7:
                    # Apply contrast
                    augmented_img = change_contrast(augmented_img, 1.2)
                elif i == 8:
                    # Apply color balance
                    augmented_img = color_balance(augmented_img, (0.8, 1.2, 1.2))
                elif i == 9:
                    # Apply blur
                    augmented_img = gaussian_blur(augmented_img, (15, 15))
                elif i == 10:
                    # Apply saturation
                    augmented_img = change_saturation(augmented_img, 1.5)
                elif i == 11:
                    augmented_img = rotate_image(augmented_img, 40)
                elif i == 12:
                    augmented_img = rotate_image(augmented_img, 50)
                elif i == 13:
                    augmented_img = rotate_image(augmented_img, 60)
                elif i == 14:
                    augmented_img = rotate_image(augmented_img, 70)
                elif i == 15:
                    augmented_img = rotate_image(augmented_img, 80)
                elif i == 16:
                    augmented_img = rotate_image(augmented_img, 90)
                elif i == 17:
                    augmented_img = rotate_image(augmented_img, 100)
                elif i == 18:
                    augmented_img = rotate_image(augmented_img, 110)
                elif i == 19:
                    augmented_img = rotate_image(augmented_img, 120)
                elif i == 20:
                    augmented_img = flip_image(augmented_img, -1)
                elif i == 21:
                    augmented_img = flip_image(augmented_img, 0)
                elif i == 22:
                    augmented_img = change_brightness(augmented_img, 40)
                elif i == 23:
                    augmented_img = change_brightness(augmented_img, 50)
                elif i == 24:
                    augmented_img = change_brightness(augmented_img, 60)
                elif i == 25:
                    augmented_img = change_brightness(augmented_img, 70)
                elif i == 26:
                    augmented_img = change_brightness(augmented_img, 80)
                elif i == 27:
                    augmented_img = change_brightness(augmented_img, 90)
                elif i == 28:
                    augmented_img = change_brightness(augmented_img, 100)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 4:
            # Diagnosticul e 4 => 7 augmentari
            for i in range(1, 8):
                augmented_img_name = f'{image_name}_aug_{i}.jpg'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                elif i == 6:
                    # Apply brightness
                    augmented_img = change_brightness(augmented_img, 30)
                elif i == 7:
                    # Apply contrast
                    augmented_img = change_contrast(augmented_img, 1.2)
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])


def process_imgs_apply_clahe(file):
    """
    Aplica clahe pe imaginile dintr-un folder
    :param file:
    :return: salveaza imaginile intr-un nou folder
    """
    input_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_aug',
                              '{}.png'.format(file))

    print(input_file)
    output_file = os.path.join(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe',
        '{}.png'.format(file))
    img = cv2.imread(input_file)
    output_img = enhancing_clahe(img)

    cv2.imwrite(output_file, cv2.resize(output_img, (512, 512)))


def process_imgs_apply_clahe_test(file):
    """
    Aplica clahe pe imaginile dintr-un folder
    :param file:
    :return: salveaza imaginile intr-un nou folder
    """
    input_file = os.path.join(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images',
                              '{}.png'.format(file))

    print(input_file)
    output_file = os.path.join(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\test_images\test_images_apply_clahe',
        '{}.png'.format(file))
    img = cv2.imread(input_file)
    output_img = enhancing_clahe(img)

    cv2.imwrite(output_file, cv2.resize(output_img, (512, 512)))


def augment_train_data_ddr_aptos_dataset():
    """
    pt diagnostic 0 - fara augmentare
    pt diagnostic 1 - augmentare (inca 5) : rotation, flipping, shearing, translation
    pt diagnostic 2 - augmentare (inca 1) : rotatie , pt jumatate din imagini
    pt diagnostic 3 - augmentare (inca 15) : rotation, flipping, shearing, translation
    pt diagnostic 4 - augmentare (inca 5): rotation, flipping, shearing, translation
    :return:
    """
    train_data = pd.read_csv(
        r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train.csv')
    train_imgs = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_images')

    new_train_data_dir = os.path.abspath(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_images_aug')
    new_train_data_csv = r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_aug.csv'
    cnt = 0
    for index, row in train_data.iterrows():  # parcurg fisierul csv
        image_name = row['id_code'].split('.')[0]
        diagnosis = row['diagnosis']

        img_path = os.path.join(train_imgs, image_name + '.png')
        img = cv2.imread(img_path)  # imaginea corespunzatoarea instantei csv

        # adaugam imaginea originala in fisier si inregistrarea in csv
        cv2.imwrite(os.path.join(new_train_data_dir, image_name + '.png'), img)
        with open(new_train_data_csv, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(
                [image_name, diagnosis])

        if diagnosis == 0:
            # Diagnosticul imaginii e 0 => fara augmentare
            pass

        elif diagnosis == 1:
            # Diagnosticul imaginii e 1 => 6 augmentari
            for i in range(1, 7):
                augmented_img_name = f'{image_name}_aug_{i}.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 2:
            # Diagnosticul e 2 => o augmentare
            cnt = cnt + 1
            if cnt % 2 == 1:
                # Apply rotation
                augmented_img_name = f'{image_name}_aug_1.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                augmented_img = rotate_image(augmented_img, 30)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 3:
            # Diagnosticul e 3 => 15 augmentari
            for i in range(1, 16):
                augmented_img_name = f'{image_name}_aug_{i}.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation - 10 rotatii
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping - 3
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                elif i == 6:
                    # Apply brightness - 8
                    augmented_img = change_brightness(augmented_img, 30)
                elif i == 7:
                    # Apply contrast
                    augmented_img = change_contrast(augmented_img, 1.2)
                elif i == 8:
                    # Apply color balance
                    augmented_img = color_balance(augmented_img, (0.8, 1.2, 1.2))
                elif i == 9:
                    # Apply blur
                    augmented_img = gaussian_blur(augmented_img, (15, 15))
                elif i == 10:
                    # Apply saturation
                    augmented_img = change_saturation(augmented_img, 1.5)
                elif i == 11:
                    augmented_img = rotate_image(augmented_img, 40)
                elif i == 12:
                    augmented_img = rotate_image(augmented_img, 50)
                elif i == 13:
                    augmented_img = rotate_image(augmented_img, 60)
                elif i == 14:
                    augmented_img = rotate_image(augmented_img, 70)
                elif i == 15:
                    augmented_img = rotate_image(augmented_img, 80)
                # Save augmented image
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
        elif diagnosis == 4:
            # Diagnosticul e 4 => 5 augmentari
            for i in range(1, 6):
                augmented_img_name = f'{image_name}_aug_{i}.png'
                augmented_img_name_csv = augmented_img_name.split('.')[0]
                augmented_img = img.copy()
                if i == 1:
                    # Apply rotation
                    augmented_img = rotate_image(augmented_img, 30)
                elif i == 2:
                    # Apply flipping
                    augmented_img = flip_image(augmented_img, 1)
                elif i == 3:
                    # Apply shearing
                    augmented_img = shear_image(augmented_img, 0.2)
                elif i == 4:
                    # Apply translation
                    augmented_img = translate_image(augmented_img, 50, -20)
                elif i == 5:
                    # Apply zooming
                    augmented_img = zoom_image(augmented_img, 1.2)
                cv2.imwrite(os.path.join(new_train_data_dir, augmented_img_name), augmented_img)
                # Update CSV with augmented image record
                with open(new_train_data_csv, mode='a', newline='') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [augmented_img_name_csv, diagnosis])
