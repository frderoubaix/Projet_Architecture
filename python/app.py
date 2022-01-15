from datetime import datetime, timedelta
import os

import csv
import psutil
import threading
import time
import tensorflow as tf


class fitThread(threading.Thread):
    def __init__(self, x_trainThread, y_trainThread):
        threading.Thread.__init__(self)
        self.x_train = x_trainThread
        self.y_train = y_trainThread
        self.etat = False  # l'état du thread est soit False (à l'arrêt)
        # soit True (en marche)

    def run(self):
        self.etat = True  # on passe en mode marche
        model.fit(self.x_train, self.y_train, epochs=5)
        self.etat = False


class evaluateThread(threading.Thread):
    def __init__(self, x_testThread, y_testThread):
        threading.Thread.__init__(self)
        self.x_test = x_testThread
        self.y_test = y_testThread
        self.etat = False  # l'état du thread est soit False (à l'arrêt)
        # soit True (en marche)

    def run(self):
        self.etat = True  # on passe en mode marche
        loss, acc = model.evaluate(self.x_test, self.y_test)
        self.etat = False


mnist = tf.keras.datasets.mnist

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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

time.sleep(10)

fitTh = fitThread(x_train, y_train)  # crée un thread de notre fit
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

evalTh = evaluateThread(x_test, y_test)
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