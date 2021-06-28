import csv
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from crossdb.postgresql.psql_trip import PsqlTrip
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class AnalysisService:
    def __init__(self, psql_trip: PsqlTrip, trip_service, database_repository, ceph_travel_repository):
        self.psql_trip = psql_trip
        self.trip_service = trip_service
        self.database = database_repository
        self.ceph_travel = ceph_travel_repository

    def load_data_for_analysis(name):
        # import the dataset
        data = pd.read_csv(r"20210527-235533_prod_driving_behavior_research.csv",
                           delimiter=',')

        datacorr = data.drop(labels=['trip_id', 'vehicle_id'], axis=1)
        datacorr.describe().to_csv("description.csv")
        print(datacorr.describe())
        data.corr().to_csv("correlation.csv")
        print(data.corr())

        features = ['travelled_distance', 'power', 'speed_limit','road_type', 'rush_hour', 'car_category',
                    'acceleration', 'composition', 'fuel_type', 'model_year',
                    'unladen_mass', 'euro_classification', 'fuel_consumption_combined',
                    'energy_label', 'top_speed',
                    'weather_comfort', 'dewPoint', 'skyInfo',
                    'daylight', 'distance', 'humidity', 'windSpeed', 'visibility',
                    'temperature']

        var_train, var_test, res_train, res_test = train_test_split(data[features], data['aggressive_behavior_combined'], test_size=0.25)
        return var_train, var_test, res_train, res_test

    def load_data_for_analysis_speeding_score(name):
        # import the dataset
        data = pd.read_csv(r"212714_prod_driving_behavior_research.csv",
                           delimiter=',')

        datacorr = data.drop(labels=['trip_id', 'vehicle_id'], axis=1)
        datacorr.describe().to_csv("100350_description.csv")
        print(datacorr.describe())
        data.corr().to_csv("100350_correlation.csv")
        print(data.corr())

        features = ['road_type', 'travelled_distance',
                    'speed_limit', 'rush_hour', 'car_category', 'power',
                    'acceleration', 'composition', 'fuel_type', 'model_year',
                    'unladen_mass', 'euro_classification', 'fuel_consumption_combined',
                    'energy_label', 'top_speed',
                    'weather_comfort', 'dewPoint', 'skyInfo',
                    'daylight', 'distance', 'humidity', 'windSpeed', 'visibility',
                    'temperature']
        var_train, var_test, res_train, res_test = train_test_split(data[features],
                                                                    data['speeding'],
                                                                    test_size=0.25)
        return var_train, var_test, res_train, res_test

    @classmethod
    def dt_analysis(cls, X_train, X_test, y_train, y_test, SMOTER = False):
        folder = ""
        if SMOTER:
            folder = "SMOTE/"
        filename = folder + 'decision_tree_result.csv'
        wtr = csv.writer(open(filename, 'w'))


        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X_train, y_train)

        y_train_predict = dtree.predict(X_train)

        y_test_predict = dtree.predict(X_test)

        cm_train = confusion_matrix(y_train, y_train_predict)
        print(cm_train)
        print(confusion_matrix(y_test, y_test_predict))
        print(classification_report(y_test, y_test_predict))

        wtr.writerows(cm_train)
        wtr.writerows(confusion_matrix(y_test, y_test_predict))
        wtr.writerow(['Score combined: ', accuracy_score(y_test, y_test_predict)])
        probas = dtree.predict_proba(X_test)
        pos_probs = probas[:, 1]
        neg_probs = probas[:, 0]
        plt.plot([0, 1], [0, 1], linestyle='--', label='No training')
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        plt.plot(fpr, tpr, marker='.', label='decision tree')
        plt.xlabel('1-specificity')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.savefig(folder + "roc_curve_pos_dt.png")
        print(dict(zip(X_train.columns, dtree.feature_importances_)))


    @classmethod
    def dt_analysis_pca(cls, X_train, X_test, y_train, y_test, SMOTER = False):
        folder = ""
        if SMOTER:
            folder = "SMOTE/"
        filename = folder + 'decision_tree_result_pca.csv'
        wtr = csv.writer(open(filename, 'w'))

        pca = PCA(n_components=7, svd_solver='randomized', whiten=True).fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X_train, y_train)
        y_train_predict = dtree.predict(X_train)
        y_test_predict = dtree.predict(X_test)

        print(confusion_matrix(y_train, y_train_predict))
        print(confusion_matrix(y_test, y_test_predict))
        print(classification_report(y_test, y_test_predict))

        wtr.writerows(confusion_matrix(y_train, y_train_predict))
        wtr.writerows(confusion_matrix(y_test, y_test_predict))
        wtr.writerow(['Score combined: ', accuracy_score(y_test, y_test_predict)])

        probas = dtree.predict_proba(X_test)
        pos_probs = probas[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        plt.plot(fpr, tpr, marker='.', label='Decision tree pca')
        plt.xlabel('1-specificity')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.savefig(folder + "roc_curve_pos_pca_dt.png")

    @classmethod
    def logistic_regression_analysis(cls, X_train, X_test, y_train, y_test, SMOTER = False):
        start_time = time.time()
        folder = ""
        if not SMOTER:
            plt.plot([0, 1], [0,1], linestyle='--', label='No training')
        else:
            folder = "SMOTE/"
        filename = folder + 'logistic_regresion_result.csv'
        wtr = csv.writer(open(filename, 'w'))
        wtr2 = csv.writer(open(folder+ 'grid_cv_result.txt', 'w'))

        #Only run following to determine the best params

        # model = LogisticRegression(max_iter=500)
        # solvers = ['newton-cg', 'lbfgs', 'liblinear']
        # penalty = ['l2']
        # c_values = [100, 10, 1.0, 0.1, 0.01]
        # grid = dict(solver=solvers, penalty=penalty, C=c_values)
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
        #                            error_score=0)
        # X = StandardScaler().fit_transform(X)
        # grid_result = grid_search.fit(X, y)
        # wtr2.writerows(grid_result.best_params_)
        #
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        # model.predict_proba(X)

        #Logistic regression model
        model = LogisticRegression(max_iter=500, solver='newton-cg', C=1.0, penalty='l2', random_state=47).fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)
        cm_train = confusion_matrix(y_train, y_train_pred)
        print(cm_train)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        wtr.writerows(cm_train)
        wtr.writerows(cm)
        cr = classification_report(y_test, y_pred)
        wtr.writerows(cr)
        label = 'logistic' if not SMOTER else 'SMOTE logistic'
        probas = model.predict_proba(X_test)
        pos_probs = probas[:,1]
        neg_probs = probas[:,0]

        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        plt.plot(fpr, tpr, marker ='.', label=label)
        plt.xlabel('1-specificity')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.savefig(folder+"roc_curve_pos.png")
        print("start.. " + str(start_time))
        print("end .. " + str(time.time()))
        print('lg Time: ' + str(time.time() - start_time) + " smote =" + str(SMOTER))




    @classmethod
    def logistic_regression_analysis_pca(cls, X_train, X_test, y_train, y_test, SMOTER = False):
        start_time = time.time()
        folder = ""
        if SMOTER:
            folder = "SMOTE/"
        filename = folder + 'logistic_regresion_pca_result.csv'
        #                           delimiter=';')
        wtr = csv.writer(open(filename, 'w'))
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        pca = PCA(n_components=7, svd_solver='randomized', whiten=True).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        model = LogisticRegression(max_iter=500, solver='newton-cg', C=1.0, penalty='l2', random_state=47).fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_pred_train)
        print(cm_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        wtr.writerows(cm_train)
        wtr.writerows(cm)
        cr = classification_report(y_test, y_pred)
        wtr.writerows(cr)

        probas = model.predict_proba(X_test)
        pos_probs = probas[:, 1]
        neg_probs = probas[:, 0]
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        plt.plot(fpr, tpr, marker='.', label='logistic pca')
        plt.xlabel('1-specificity')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.savefig(folder + "roc_curve_pos_pca.png")
        print("start.. " + str(start_time))
        print("end .. " + str(time.time()))
        print('lg Time pca: ' + str(time.time() - start_time) + " smote =" + str(SMOTER))


    @classmethod
    def Smote(cls, X_train, X_test, y_train, y_test):
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        X_test, y_test = oversample.fit_resample(X_test, y_test)
        return X_train, X_test, y_train, y_test

    @classmethod
    def dataStats(cls, data):
        #used for comparison between data sets
        current_time = time.strftime("%H%M%S")
        plt.hist(data['aggressive_behavior_combined'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time+"aggresive_behavior_combined_hist.png")
        plt.close()
        plt.hist(data['road_type'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "road_type_hist.png")
        plt.close()
        plt.hist(data['travelled_distance'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "travelled_distance_hist.png")
        plt.close()
        plt.hist(data['aggressive_behavior_combined'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "aggresive_behavior_combined_hist.png")
        plt.close()
        plt.hist(data['speed_limit'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "speed_limit_hist.png")
        plt.close()
        plt.hist(data['rush_hour'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "rush_hour_hist.png")
        plt.close()
        plt.hist(data['car_category'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "car_category_hist.png")
        plt.close()
        plt.hist(data['power'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "power_hist.png")
        plt.close()
        plt.hist(data['acceleration'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "acceleration_hist.png")
        plt.close()
        plt.hist(data['composition'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "composition_hist.png")
        plt.close()
        plt.hist(data['fuel_type'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "fuel_type_hist.png")
        plt.close()
        plt.hist(data['model_year'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "model_year_hist.png")
        plt.close()
        plt.hist(data['unladen_mass'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "unladen_mass_hist.png")
        plt.close()

        plt.hist(data['euro_classification'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "euro_classification_hist.png")
        plt.close()
        plt.hist(data['fuel_consumption_combined'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "fuel_consumption_combined_hist.png")
        plt.close()
        plt.hist(data['energy_label'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "energy_label_hist.png")
        plt.close()
        plt.hist(data['top_speed'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "top_speed_hist.png")
        plt.close()
        plt.hist(data['weather_comfort'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "weather_comfort_hist.png")
        plt.close()

        plt.hist(data['dewPoint'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "dewPoint_hist.png")
        plt.close()

        plt.hist(data['skyInfo'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "skyInfo_hist.png")
        plt.close()
        plt.hist(data['daylight'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "daylight_hist.png")
        plt.close()

        plt.hist(data['distance'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "distance_hist.png")
        plt.close()

        plt.hist(data['humidity'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "humidity_hist.png")
        plt.close()

        plt.hist(data['windSpeed'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "windSpeed_hist.png")
        plt.close()

        plt.hist(data['visibility'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "visibility_hist.png")
        plt.close()

        plt.hist(data['temperature'])
        plt.title("Mean")
        plt.xlabel('value')
        plt.ylabel("fequency")
        plt.savefig(current_time + "temperature_hist.png")

    @classmethod
    def speeding_score_analysis(cls, X, y):
        folder = "/"
        filename = folder + 'logistic_regresion_pca_speeding_score_result.csv'
        #                           delimiter=';')
        wtr = csv.writer(open(filename, 'w'))
        X = StandardScaler().fit_transform(X)

        pca = PCA(n_components=7, svd_solver='randomized', whiten=True).fit(X)
        X = pca.transform(X)

        var_train, var_test, res_train, res_test = train_test_split(X, y, test_size=0.25)
        model = LogisticRegression(max_iter=500, solver='newton-cg', C=1.0, penalty='l2', random_state=47).fit(
            var_train, res_train)
        # model.predict_proba(X)

        y_pred = model.predict(var_test)
        cm = confusion_matrix(res_test, y_pred)
        wtr.writerows(cm)
        cr = classification_report(res_test, y_pred)
        wtr.writerows(cr)