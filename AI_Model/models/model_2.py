import os.path
from keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from data_processing import data_processing

train_data, test_data = data_processing()

EPOCHS = 3
DETAILS = 'augmented imgs - added layers'
VERSION = '4'

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

# augmentarea datelor
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

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
# Stratul 1: Strat de convolutie 2D
model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 3)))

# Stratul 2: Strat de normalizare a batch-ului
model.add(BatchNormalization())

# Stratul 3: Strat de activare ReLU
model.add(Activation('relu'))

# Stratul 4: Strat de pooling maxim
model.add(MaxPooling2D(pool_size=(2, 2)))

# Stratul 5: Strat de convolutie 2D
model.add(Conv2D(64, (3, 3)))

# Stratul 6: Strat de normalizare a batch-ului
model.add(BatchNormalization())

# Stratul 7: Strat de activare ReLU
model.add(Activation('relu'))

# Stratul 8: Strat de pooling maxim
model.add(MaxPooling2D(pool_size=(2, 2)))

# Stratul 9: Strat de convolutie 2D
model.add(Conv2D(96, (3, 3)))

# Stratul 10: Strat de normalizare a batch-ului
model.add(BatchNormalization())

# Stratul 11: Strat de activare ReLU
model.add(Activation('relu'))

# Stratul 12: Strat de pooling maxim
model.add(MaxPooling2D(pool_size=(2, 2)))

# Stratul 13: Strat de aplatizare
model.add(Flatten())

# Stratul 14: Strat de conectare completă (FC)
model.add(Dense(1000))

# Stratul 15: Strat de normalizare a batch-ului
model.add(BatchNormalization())

# Stratul 16: Strat de activare ReLU
model.add(Activation('relu'))

# Stratul 17: Strat de conectare completă (FC)
model.add(Dense(500))

# Stratul 18: Strat de normalizare a batch-ului
model.add(BatchNormalization())

# Stratul 19: Strat de activare ReLU
model.add(Activation('relu'))

# Stratul 20: Strat de conectare completă (FC) cu activare SoftMax
model.add(Dense(5, activation='softmax'))

model.summary()