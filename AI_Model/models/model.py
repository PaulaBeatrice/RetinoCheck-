import csv
import os.path
from datetime import datetime
import numpy as np
import pandas as pd
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from data_processing import data_processing

train_data, test_data = data_processing()

EPOCHS = 1
DETAILS = 'no filter images'
VERSION = '1'

invalid_imgs = train_data[train_data['file_path'].apply(lambda x: not os.path.exists(x))]
print("Nr img invalide: " + str(len(invalid_imgs)))
if len(invalid_imgs) > 0:
    for idx, row in invalid_imgs.iterrows():
        print(row['file_path'])

    # excludem img invalide din train_data
    train_data = train_data[~train_data['file_path'].isin(invalid_imgs['file_path'])]
else:
    print("Nu exista img invalide")

invalid_imgs_test = test_data[test_data['file_path'].apply(lambda x: not os.path.exists(x))]
print("Nr img invalide: " + str(len(invalid_imgs_test)))
if len(invalid_imgs_test) > 0:
    for idx, row in invalid_imgs_test.iterrows():
        print(row['file_path'])

    # excludem img invalide din train_data
    test_data = test_data[~test_data['file_path'].isin(invalid_imgs['file_path'])]
else:
    print("Nu exista img- test invalide")

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=None,
    x_col='file_path',
    y_col='diagnosis',
    target_size=(512, 512),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=None,
    x_col='file_path',
    y_col='diagnosis',
    target_size=(512, 512),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# antrenare model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=EPOCHS
)

# evaluare model
test_generator = datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=None,
    x_col='file_path',
    y_col='diagnosis',
    target_size=(512, 512),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

test_data['diagnosis'] = test_data['diagnosis'].astype(str)

if test_generator.samples > 0:
    test_loss, test_accuracy = model.evaluate_generator(test_generator, steps=len(test_generator))
    print('Test accuracy EVALUATE : ', test_accuracy)

    correct_predicted = 0
    folder_imgs_path = '/data/APTOS/test_images/test_images'

    test_data_csv = pd.read_csv('/data/APTOS/test.csv')

    true_labels = []
    predicted_labels = []

    # parcurge folder, ia fiecare imagine, predictie si verificam corectitudinea
    for img_filename in os.listdir(folder_imgs_path):
        img_path = os.path.join(folder_imgs_path, img_filename)
        img_name = os.path.splitext(img_filename)[0]
        img = image.load_img(img_path, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict_generator(img_array)
        predicted_class = np.argmax(predictions[0])

        test_row = test_data_csv[test_data_csv['id_code'] == img_name]
        correct_label = int(test_row['diagnosis'])

        true_labels.append(correct_label)
        predicted_labels.append(predicted_class)

        if correct_label == predicted_class:
            correct_predicted += 1
            print(img_path)

    acc = correct_predicted / len(os.listdir(folder_imgs_path))
    print("Manual Accuracy : " + str(acc))



    correct_predicted_tr = 0
    folder_imgs_path_tr = '/data/APTOS/train_images/train_images'

    train_data_csv = pd.read_csv('/data/APTOS/train_1.csv')

    true_labels = []
    predicted_labels = []

    # parcurge folder, ia fiecare imagine, predictie si verificam corectitudinea
    for img_filename in os.listdir(folder_imgs_path_tr):
        img_path = os.path.join(folder_imgs_path_tr, img_filename)
        img_name = os.path.splitext(img_filename)[0]
        img = image.load_img(img_path, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        test_row = train_data_csv[train_data_csv['id_code'] == img_name]
        correct_label = int(test_row['diagnosis'])

        true_labels.append(correct_label)
        predicted_labels.append(predicted_class)

        if correct_label == predicted_class:
            correct_predicted_tr += 1
            print(img_path)

    acc_train = correct_predicted_tr / len(os.listdir(folder_imgs_path_tr))
    print("Manual Accuracy Training Data : " + str(acc_train))




    # Calcularea matricei de confuzie folosind rezultatele manuale
    cm_manual = confusion_matrix(true_labels, predicted_labels)

    results_file_path = '/results.csv'

    with open(results_file_path, mode='r') as results_file:
        results_reader = csv.reader(results_file)
        num_records = sum(1 for _ in results_reader)

    # Afișarea matricei de confuzie
    class_label = ['No DR - 0', 'Mild - 1', 'Moderate - 2', 'Severe - 3', 'Proliferative DR - 4']
    disp_manual = ConfusionMatrixDisplay(confusion_matrix=cm_manual, display_labels=class_label)
    disp_manual.plot(cmap='Blues', values_format='d')

    # Salvarea matricei de confuzie
    confusion_matrix_image_path = '/data/img_results\\'
    confusion_matrix_image_name = f"{num_records}.jpg"
    confusion_matrix_image_full_path = os.path.join(confusion_matrix_image_path, confusion_matrix_image_name)
    plt.savefig(confusion_matrix_image_full_path)
    plt.close()
    print(f"Confusion matrix image saved at: {confusion_matrix_image_full_path}")

    plt.show()

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(results_file_path, mode='a', newline='') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(
            [num_records, current_date, acc, test_accuracy, correct_predicted, EPOCHS, DETAILS, VERSION, acc_train])

else:
    print("Nu sunt img valide in generator")

class_label = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

while True:
    img_path = input("Enter the path to the image: ")

    if img_path == '0':
        print('Exit')
        break
    else:
        img = image.load_img(img_path, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        predicted_class_name = class_label[predicted_class]
        print("Predicted class = " + predicted_class_name)
        print(predictions)