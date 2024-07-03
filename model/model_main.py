# Import biblotek
import os
from datetime import datetime
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import Input

# Stałe wartosci
BASE_PATH = './data/spectograms/'
INPUT_SHAPE = (224,224)

class LungModel:

    model = None
    spectogram_paths = []
    spectogram_labels = []
    number_of_classes = 0
    class_names = []
    train_dataset = None
    test_dataset = None
    train_history = None
    _y_test = None

    #Wczytanie sciezek plików oraz ich labelów
    def load_data_paths_and_labels(self):

        self.class_names = os.listdir(f'{BASE_PATH}')
        print(f"CLASS NAMES: {self.class_names}")

        for class_number, name in enumerate(self.class_names):
            spect_files = os.listdir(f"{BASE_PATH}/{name}")

            for spectogram in spect_files:
                img_path = os.path.join(BASE_PATH, name, spectogram)
                self.spectogram_paths.append(img_path)
                self.spectogram_labels.append(class_number)

        self.number_of_classes = len(self.class_names)


    # Wczytywanie obrazów
    def load_and_preprocess_image(self, file_path, label_):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0

        label = tf.keras.utils.to_categorical(label_, num_classes=self.number_of_classes)

        return img, label

    #Przygotowanie danych
    def prepare_dataset(self,_batch_size = 1):

        # Podział zbioru na dane treningowe oraz testowe
        X_train, X_test, y_train, y_test = train_test_split(self.spectogram_paths, self.spectogram_labels, test_size=0.3, random_state=42)
        self._y_test = y_test

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        #Mapowanie po obrazach
        self.train_dataset = self.train_dataset.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #Batchowanie oraz prefetch danych
        self.train_dataset = self.train_dataset.batch(_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.batch(_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        # Sprawdzenie rozmiarów zbiorów danych
        print("Rozmiar zbioru treningowego:",  tf.data.experimental.cardinality(self.train_dataset).numpy())
        print("Rozmiar zbioru testowego:", tf.data.experimental.cardinality(self.test_dataset).numpy())
        print("Ilość rozpatrywanych klas : ", len(self.class_names))


    # Trenowanie modelu
    def train_model(self, _train_epochs = 1):
        #print("Kształt zbioru treningowego:", self.train_dataset.element_spec)
        #print("Kształt zbioru testowego:", self.test_dataset.element_spec)
        print("Trenowanie modelu...")

        self.train_history = self.model.fit(
            self.train_dataset,
            epochs = _train_epochs,
            validation_data = self.test_dataset)

    #Tworzenie modelu
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add( Input(shape = ( INPUT_SHAPE[0], INPUT_SHAPE[1], 3))) # Warstwa wejściowa
        model.add(layers.Conv2D(8, (3, 3), padding='same',activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(16, (3, 3), padding='same',activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units = self.number_of_classes, activation='softmax')) # Warstwa wyjściowa

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
        )

        self.model = model

    # Pokaż rezultaty treningu
    def show_results(self):
        hist = self.train_history

        plt.figure(figsize=(15,12))
        # Accuracy
        plt.subplot(2,2,1)
        plt.title("Accuracy")
        plt.plot(hist.history['accuracy'],'r',label="accuracy")
        plt.plot(hist.history['val_accuracy'],'b',label="val_accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(2,2,2)
        plt.title("Loss")
        plt.plot(hist.history['loss'],'r',label="loss")
        plt.plot(hist.history['val_loss'],'b',label="val_loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()

        # Precision
        plt.subplot(2,2,3)
        plt.title("Precision")
        plt.plot(hist.history['precision'],'r',label="precision")
        plt.plot(hist.history['val_precision'],'b',label="val_precision")
        plt.xlabel("Epoch #")
        plt.ylabel("Precision")
        plt.legend()

        # Recall
        plt.subplot(2,2,4)
        plt.title("Recall")
        plt.plot(hist.history['recall'],'r',label="recall")
        plt.plot(hist.history['val_recall'],'b',label="val_recall")
        plt.xlabel("Epoch #")
        plt.ylabel("Recall")
        plt.legend()

        #Create directory for save results
        if not os.path.exists('./results'):
            os.makedirs('./results')

        # Save plot
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y-%H_%M")
        plt.savefig(f'./results/res_{current_time}.png')
            
        plt.show()


    def test_model(self):
        print("Testowanie modelu...")

        y_pred_ = self.model.predict(self.test_dataset)

        test_acc = sum(self._y_test == np.argmax(y_pred_, axis= 1)) / len(self._y_test)

        print(f"Accuracy model: {np.round(test_acc * 100,2)} %")

        return y_pred_, test_acc


    def show_confusion_matrix(self, y_pred_, class_names):
        confusion_mtx = tf.math.confusion_matrix(self._y_test, np.argmax(y_pred_, axis=1))

        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_mtx,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        
        # Save confusion matrix
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y-%H_%M")
        plt.savefig(f'./results/confMatrix_{current_time}.png')

        plt.show()

    def save_model(self,result_accurate = 0):
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y-%H_%M")
        self.model.save(f"./models/model_{current_time}__{TRAIN_EPOCHS}__{result_accurate:.04f}.keras")
        print("Model saved...")


if __name__ == "__main__":

    #Tworzenie obiektu modelu
    lung_disease_model = LungModel()

    #Wczytanie ścieżek plików oraz etykiet
    lung_disease_model.load_data_paths_and_labels()

    #Tworzenie modelu
    lung_disease_model.create_model()

    # Wyświetalanie podsumowania modelu
    # lung_disease_model.model.summary()

    #Przygotowanie danych
    BATCH_SIZE = 16
    lung_disease_model.prepare_dataset(BATCH_SIZE)

    # Trenowanie modelu
    TRAIN_EPOCHS = 10
    lung_disease_model.train_model(TRAIN_EPOCHS)

    # Wyświetlanie wyników treningu
    lung_disease_model.show_results()

    # Ewaluacja modelu
    y_pred, result_acc = lung_disease_model.test_model()

    # Wyświetlenie confision_matrix
    lung_disease_model.show_confusion_matrix(y_pred,lung_disease_model.class_names)

    # Tworzenie folderu do zapisu modelu
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Zapis modelu
    lung_disease_model.save_model(result_acc)
