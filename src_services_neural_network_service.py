import csv

import matplotlib.pyplot as plt
import tensorflow as tf
from crossdb.postgresql.psql_trip import PsqlTrip
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras


class NeuralNetworkService:
    def __init__(self, psql_trip: PsqlTrip, trip_service, database_repository, ceph_travel_repository):
        self.psql_trip = psql_trip
        self.trip_service = trip_service
        self.database = database_repository
        self.ceph_travel = ceph_travel_repository


    def create_model(self):
        return keras.Sequential([
            keras.layers.Flatten(input_shape=(24,)),
            keras.layers.Dense(512, activation=tf.nn.sigmoid),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ])



    @classmethod
    def neural_network_analysis(cls, X_train, X_test, y_train, y_test, filename):
        wtr = csv.writer(open(filename + ".csv", 'w'))
        model = cls.create_model(cls)
        model.compile(optimizer='Adam', loss = 'mean_absolute_error', metrics=['accuracy'])
        epochs = 200
        print(model.summary())
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=3000)
        test_loss, test_acc, *catch_all = model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)
        #calculate specificity
        predictions_train = model.predict(X_train)
        predictions = model.predict(X_test)
        #converting predictions to label
        pred_train = list()
        for i in range(len(predictions_train)):
            pred_train.append(1 if predictions_train[i] > 0.5 else 0)
        pred = list()
        for i in range(len(predictions)):
            pred.append(1 if predictions[i] > 0.5 else 0)



        a = accuracy_score(y_test, pred)
        print("accuracy is: ", a*100)

        cm_train = confusion_matrix(y_train, pred_train)
        print(cm_train)
        cm = confusion_matrix(y_test, pred)
        print(cm)
        # wtr.writerows(cm_train)
        # wtr.writerows(cm)
        specificity = cls.specificity(cls, y_test, pred)
        wtr.writerow([specificity])

        plt.plot(history.history['loss'])
       # plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(filename +'nn_loss.png')

        plt.close()
        plt.plot(history.history['accuracy'])

        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(filename +'nn_accuracy.png')

    def specificity(self, y_true, y_pred):
        #pls note!.. positive and negative might be switched here!
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn/(tn+fp)

