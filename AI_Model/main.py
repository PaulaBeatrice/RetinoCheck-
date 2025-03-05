import os.path
import os
import cv2
from skimage.exposure import rescale_intensity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def show_distribution():
    csv_path = r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\DDR\DR_aug.csv'
    data = pd.read_csv(csv_path)

    # Numărul total de imagini
    total_images = len(data)

    # Numărarea diagnosticelor
    diagnosis_counts = data['diagnosis'].value_counts().sort_index()

    # Calcularea procentelor
    percentages = (diagnosis_counts / total_images) * 100

    # Afisarea rezultatelor
    for diagnosis, count, percentage in zip(diagnosis_counts.index, diagnosis_counts.values, percentages):
        print(f"Diagnosticul {diagnosis}: Număr = {count}, Procent = {percentage:.2f}%")

    print(f"Număr total de imagini: {total_images}")


def show_distribution_aptos_ddr():
    csv_path = r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train.csv'
    data = pd.read_csv(csv_path)

    # Numărul total de imagini
    total_images = len(data)

    # Numărarea diagnosticelor
    diagnosis_counts = data['diagnosis'].value_counts().sort_index()

    # Calcularea procentelor
    percentages = (diagnosis_counts / total_images) * 100

    # Afisarea rezultatelor
    for diagnosis, count, percentage in zip(diagnosis_counts.index, diagnosis_counts.values, percentages):
        print(f"Diagnosticul {diagnosis}: Număr = {count}, Procent = {percentage:.2f}%")

    print(f"Număr total de imagini: {total_images}")


def show_distribution_aptos_ddr_aug():
    csv_path = r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_aug.csv'
    data = pd.read_csv(csv_path)

    # Numărul total de imagini
    total_images = len(data)

    # Numărarea diagnosticelor
    diagnosis_counts = data['diagnosis'].value_counts().sort_index()

    # Calcularea procentelor
    percentages = (diagnosis_counts / total_images) * 100

    # Afisarea rezultatelor
    for diagnosis, count, percentage in zip(diagnosis_counts.index, diagnosis_counts.values, percentages):
        print(f"Diagnosticul {diagnosis}: Număr = {count}, Procent = {percentage:.2f}%")

    print(f"Număr total de imagini: {total_images}")


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


def apply_clahe_filter(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)

            output_img = enhancing_clahe(image)

            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, cv2.resize(output_img, (512, 512)))


def apply_gaussian_filter(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)

            # output_img = ski.filters.gaussian(image, sigma=1)  # Apply Gaussian filter
            # output_img = apply_gaussian_filter(image)
            output_img = cv2.GaussianBlur(image, (15, 15), 0)

            # Save the output image
            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, cv2.resize(output_img, (512, 512)))


def normalise(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)
            output_img = rescale_intensity(image, in_range='image', out_range=(0, 255)).astype(np.uint8)
            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, cv2.resize(output_img, (512, 512)))


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
            return img
        else:
            img1 = img[:, :, 0][
                np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)

            img = crop_Image(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, depth = img.shape
            x = int(width / 2)
            y = int(height / 2)
            r = np.amin((x, y))

            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
            img = cv2.bitwise_and(img, img, mask=circle_img)
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128)
            output_img = img

            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, cv2.resize(output_img, (512, 512)))


def plot_class_distribution(csv_file_path, title):
    data = pd.read_csv(csv_file_path)

    class_counts = data['diagnosis'].value_counts()

    class_labels = {0: 'Fără retinopatie', 1: 'Formă ușoară', 2: 'Formă moderată', 3: 'Formă severă',
                    4: 'Formă proliferativă'}

    class_distribution = [(class_labels[k], v) for k, v in class_counts.items()]

    classes, sizes = zip(*class_distribution)

    colors = sns.color_palette("rocket")
    plt.pie(sizes, labels=classes, autopct=lambda p: '{:.1f}%'.format(p), colors=colors)

    plt.subplots_adjust(top=0.85)

    plt.title(title, y=1.05)

    plt.show()


if __name__ == '__main__':
    # d_train, d_test = data_processing()
    # d_train, d_test = data_processing_aug()
    #
    # # afisare distributie output train si test data
    # # graphics_distribution(d_train, "Train Data")
    # # graphics_distribution(d_test, "Test Data")
    # # resize_all_imgs(6, list(d_train.id_code.values))
    # # view_imgs_enhanced(d_train, 5, color_scale=None)
    #
    # # multi_process_ims_v2(6, list(d_train.id_code.values))
    #
    # # multi_process_ims_test(6, list(d_test.id_code.values))
    # csv_path = 'D:\\FACULTATE\\FACULTATE\\LICENTA\\proiect\\data\\APTOS\\train_1.csv'
    # # folder_imgs_path = 'D:\\FACULTATE\\FACULTATE\\LICENTA\\proiect\\data\\APTOS\\test_images\\test_images'
    # #
    # # # parcurge folder, ia fiecare imagine, predictie si verificam corectitudinea
    # # for img_filename in os.listdir(folder_imgs_path):
    # #     img_path = os.path.join(folder_imgs_path, img_filename)
    # #     img_name = os.path.splitext(img_filename)[0]
    # #     # print(img_name[0])
    #
    # # print( len(os.listdir(folder_imgs_path)) )
    #
    # data = pd.read_csv(csv_path)
    #
    # # Numărul total de imagini
    # total_images = len(data)
    #
    # # Numărarea diagnosticelor
    # diagnosis_counts = data['diagnosis'].value_counts().sort_index()
    #
    # # Calcularea procentelor
    # percentages = (diagnosis_counts / total_images) * 100
    #
    # # Afisarea rezultatelor
    # for diagnosis, count, percentage in zip(diagnosis_counts.index, diagnosis_counts.values, percentages):
    #     print(f"Diagnosticul {diagnosis}: Număr = {count}, Procent = {percentage:.2f}%")
    #
    # print(f"Număr total de imagini: {total_images}")

    # show_distribution_aptos_ddr()
    # augment_train_data_ddr_aptos_dataset()
    # show_distribution_aptos_ddr_aug()
    # apply_clahe_filter(
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_images_aug',
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS_DDR\train_images_aug_clahe')
    # apply_gaussian_filter(
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe',
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe_gaussian_filter')
    # normalise(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe',
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe_normalise')
    # circle_crop(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe',
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe_crop') circle_crop(
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe_gaussian_filter',
    # r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_clahe_gaussian_crop')
    # circle_crop(r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe_normalise
    # ', r'D:\FACULTATE\FACULTATE\LICENTA\proiect\data\APTOS\train_images\train_images_apply_clahe_normalisez_crop')
    csv_file_path = "D:\\FACULTATE\\FACULTATE\\LICENTA\\proiect\\data\\APTOS\\train_1.csv"
    plot_class_distribution(csv_file_path, "Distribuția setului de date APTOS 2019")
    csv_file_path = "D:\\FACULTATE\\FACULTATE\\LICENTA\\proiect\\data\\DDR\\DR_grading.csv"
    plot_class_distribution(csv_file_path, "Distribuția setului de date DDR")
