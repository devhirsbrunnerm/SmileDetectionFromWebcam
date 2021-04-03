import os
from os import listdir
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications import VGG16
from keras.layers import Dense, GlobalMaxPooling2D, Input, Dropout, Activation, BatchNormalization, Flatten
from keras import Model
from keras.models import load_model


class SmileDetectionModel:
    def __init__(self, train_model = False):
        print('Initializing Smile detection model...')
        self.data = []
        self.labels = []
        self.categorical_labels = []
        self.le = None

        
        # if specified that the model should be retrained
        if train_model:
            self.__create_model()
        else:
            self.le = self.le = LabelEncoder().fit(['Laecheln', 'Nicht-Laecheln'])
            # try to load the model, if the model is not found, train it from scratch.
            try:
                dirname = os.path.dirname(__file__) 
                model_path = os.path.join(dirname, './saved_model') 
                self.model = load_model(model_path)
            except Exception as e:
                print('Could not load model (this is normal when running program for the first time), training it from scratch')
                self.__create_model()

        print('finished initializing Smile detection model')

    def __create_model(self):
        self.__load_data()

        # Standardize the pixel values from 0 - 255 to 0 - 1
        self.data = np.array(self.data, dtype = "float") / 255.0
        self.labels = np.array(self.labels)

        self.__encode_labels()
        self.model = self.__generate_model()
        self.__train_model()

    def __encode_labels(self):
        self.le = LabelEncoder().fit(self.labels)
        # convert the labels of the dataset to numeric ones and convert these to a binary distribution 
        self.categorical_labels = to_categorical(self.le.transform(self.labels), len(self.le.classes_))

    def __load_data(self):
        dirname = os.path.dirname(__file__) 
        base_path = os.path.join(dirname, './training_data')
        print('Loading data for positives') 
        self.__load_data_for_label(f'{base_path}/positives', 'Laecheln')
        print('Loading data for negatives')
        self.__load_data_for_label(f'{base_path}/negatives', 'Nicht-Laecheln')

    def __generate_model(self):
        input_shape = (64, 64, 3)
        # Load the VGG16 model and do not include the top
        pretrained_model = VGG16(input_shape=input_shape, 
                         include_top=False, 
                         weights="imagenet",
                         )
        
        # Set every layer (except for the last 4) of the VGG16 to not trainable
        for layer in pretrained_model.layers[:-4]:
            layer.trainable = False

        # Get the last output where we will add our own layers
        last_layer = pretrained_model.get_layer('block5_pool')
        last_output = last_layer.output
            
        # Own layers for this specific use case

        # Convert CNN to fully connected layers
        x = Flatten()(last_output)
        x = Dense(512, activation='relu')(x)

        # Add a layer to handle batch training (used during training the model)
        x = BatchNormalization()(x)

        # prevent overfitting
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)

        # Add a dense layer with as many outputs as there are classes (2)
        x = Dense(len(self.le.classes_))(x)

        # Get a probability distribution for the classes 
        x = Activation("softmax")(x)

        model = Model(pretrained_model.input, x)
        
        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=['accuracy'])

        print('Model summary:')
        model.summary()
        return model
    
    def __train_model(self):
        (trainX, testX, trainY, testY) = train_test_split(self.data, self.categorical_labels, stratify=self.categorical_labels,
            test_size = 0.20, random_state = 4)
        
        print(f'Training Model with {len(trainX)} training data and {len(testX)} test data')
        self.model.fit(trainX, trainY, validation_data = (testX, testY),
            batch_size = 64, epochs = 15, verbose = 1)

        dirname = os.path.dirname(__file__) 
        model_path = os.path.join(dirname, './saved_model') 
        self.model.save(model_path)

    def __load_data_for_label(self, path, label):
        for i, file in enumerate(listdir(path)):
            # Load the grayscale images and resize them
            image = cv2.imread(f'{path}/{file}', flags=cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)
            # Fill three channels with the grayscale values, this is needed because of the architecture of VGG16
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            self.data.append(image)
            self.labels.append(label)

    def predict(self, image):
        # Resize image to fit Network architecture
        image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)

        # Fill three channels with the grayscale values
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        # Standardize pixel values to 0 - 1
        image = np.array(image, dtype = "float") / 255.0

        # Because normally one does Batch processing, you pass an array with images to classify, this is a single image so wrap it into an array (add a dimension before all others)
        image = np.expand_dims(image, axis = 0)
        categorical = self.model.predict(image)
        # Read the highest probability and get the label text with the label encoder.
        result = self.le.inverse_transform(
                [
                    np.argmax(categorical)
                ]
            )[0]

        return result
        

        