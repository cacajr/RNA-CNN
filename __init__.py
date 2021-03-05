from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation

from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import backend
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# CNN NA ARQUITETURA LeNet5
class CNN_LeNet5(object):
    '''
    layers:
    INPUT =&gt; CONV =&gt; POOL =&gt; CONV =&gt; POOL =&gt; FC =&gt; FC =&gt; OUTPUT
    '''

    @staticmethod
    def build(width, height, channels, classes):
        '''
        Constroi uma CNN com arquitetura LeNet5.
 
        :param width: Largura em pixel da imagem.
        :param height: Altura em pixel da imagem.
        :param channels: Quantidade de canais da imagem.
        :param classes: Quantidade de classes para o output.
        :return: CNN do tipo LeNet5.
        '''
        inputShape = (height, width, channels)
 
        # Convolution 1
        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))

        # Max pool 1
        model.add(MaxPooling2D((2,2)))

        # Convolution 2
        model.add(Conv2D(16, (5, 5)))
        model.add(Activation("relu"))

        # Max pool 2
        model.add(MaxPooling2D((2,2)))

        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        model.add(Dense(84))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        return model

# IMPORTAR E NORMALIZAR O DATASETs: MNIST(mnist_784) CIFAR-10(cifar_10)
dataset_name = "mnist_784"
dataset = fetch_openml(dataset_name)
labels = dataset.target
data = dataset.data.astype("float32") / 255.0

# converter as imagens de 1D para o formato (28x28x1) se mnist_784 e para (32x32x3) se cifar_10
if backend.image_data_format() == "channels_last":
    if dataset_name == "mnist_784":
        data = data.reshape((data.shape[0], 28, 28, 1))
    elif dataset_name == "cifar_10":
        data = data.reshape((data.shape[0], 32, 32, 3))
else:
    if dataset_name == "mnist_784":
        data = data.reshape((data.shape[0], 1, 28, 28))
    elif dataset_name == "cifar_10":
        data = data.reshape((data.shape[0], 32, 32, 3))

# dividir o dataset entre train (75%) e test (25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels)

# Transformar labels em vetores binarios
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# INICIALIZAR E OTIMIZAR MODELO
print("[INFO] inicializando e otimizando a CNN...")
if dataset_name == "mnist_784":
    model = CNN_LeNet5.build(28, 28, 1, 10)
elif dataset_name == "cifar_10":
    model = CNN_LeNet5.build(32, 32, 3, 10)

model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])

# TREINAR A CNN
print("[INFO] treinando a CNN...")
H = model.fit(trainX, trainY, batch_size=128, epochs=20, verbose=2,
          validation_data=(testX, testY))

# AVALIACAO CNN
print("[INFO] avaliando a CNN...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(label) for label in range(10)]))

# plotar loss e accuracy para os datasets 'train' e 'test'
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('cnn.png', bbox_inches='tight')