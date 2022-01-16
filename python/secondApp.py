from datetime import datetime, timedelta
import os

import csv
import psutil
import threading
import time
import tensorflow as tf
from tensorflow import keras


class fitThread(threading.Thread):
    def __init__(self, x_train_images, y_train_labels):
        threading.Thread.__init__(self)
        self.x_train_images = x_train_images
        self.y_train_labels = y_train_labels
        self.etat = False  # l'état du thread est soit False (à l'arrêt)
        # soit True (en marche)

    def run(self):
        self.etat = True  # on passe en mode marche
        model.fit(self.x_train_images, self.y_train_labels, epochs=5)
        self.etat = False


class evaluateThread(threading.Thread):
    def __init__(self, x_test_images, y_test_labels):
        threading.Thread.__init__(self)
        self.x_test_images = x_test_images
        self.y_test_labels = y_test_labels
        self.etat = False  # l'état du thread est soit False (à l'arrêt)
        # soit True (en marche)

    def run(self):
        self.etat = True  # on passe en mode marche
        loss, acc = model.evaluate(self.x_test_images, self.y_test_labels)
        self.etat = False

pid = 0

header = ['time', 'memory', 'cpu']

for proc in psutil.process_iter():
    if proc.pid.__str__().__eq__(os.getpid().__str__()):
        pid = proc.pid

process = psutil.Process(pid)

f = open('data.csv', 'w', newline='', encoding='UTF8')
writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

# write the header
writer.writerow(header)

# close data
f.close()

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))


model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, name='Dense')
])
model.summary()

testing = False


model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

data_zero = [(datetime.now() - timedelta(hours=1)), 0, 0]

f = open('data.csv', 'a', newline='', encoding='UTF8')
writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(data_zero)
f.close()

debutFit = round(time.time() * 1000)

fitTh = fitThread(train_images, train_labels)  # crée un thread de notre fit
fitTh.start()  # démarre le thread,

while fitTh.etat:
    # on attend que le thread s'arrête
    # il faut introduire l'instruction time.sleep pour temporiser, il n'est pas
    # nécessaire de vérifier sans cesse que le thread est toujours en marche
    # il suffit de le vérifier tous les x millisecondes
    # dans le cas contraire, la machine passe son temps à vérifier au lieu
    # de se consacrer à l'exécution du thread
    fl = (float(process.memory_percent())).__str__()
    cpuPercent = (float(process.cpu_percent())).__str__()
    data = [(datetime.now() - timedelta(hours=1)), fl, cpuPercent]
    f = open('data.csv', 'a', newline='', encoding='UTF8')
    writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(data)
    f.close()
    time.sleep(0.1)

finFit = round(time.time() * 1000)

execFit = finFit - debutFit
print("Fit duration in ms : " + execFit.__str__)

data_zero = [(datetime.now() - timedelta(hours=1)), 0, 0]

f = open('data.csv', 'a', newline='', encoding='UTF8')
writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(data_zero)
f.close()

# temps d'arrêt entre fit et evaluate
time.sleep(10)

data_zero = [(datetime.now() - timedelta(hours=1)), 0, 0]

f = open('data.csv', 'a', newline='', encoding='UTF8')
writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(data_zero)
f.close()

debutEval = round(time.time() * 1000)

evalTh = evaluateThread(test_images, test_labels)
evalTh.start()

while evalTh.etat:
    fl = (float(process.memory_percent())).__str__()
    cpuPercent = (float(process.cpu_percent())).__str__()
    data = [(datetime.now() - timedelta(hours=1)), fl, cpuPercent]
    f = open('data.csv', 'a', newline='', encoding='UTF8')
    writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(data)
    f.close()
    time.sleep(0.1)

finEval = round(time.time() * 1000)

execEval = finEval - debutEval
print("Eval duration in ms : " + execEval.__str__)