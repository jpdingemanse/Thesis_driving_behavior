import time

from crosslogging.logger import Logger
from src.containers.services import Services

# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator

import logging
import urllib3
import datetime as dt

from src.services.analysis_service import AnalysisService
from src.services.neural_network_service import NeuralNetworkService

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
Logger.format_logging_standard(level=logging.WARNING)
DataService = Services.data_service()

if __name__ == "__main__":
    name = None
    # Done run them all at the same run because of processing power
    name = DataService.get_all_data(2017)
    # name = DataService.get_all_data(2018)
    # name = DataService.get_all_data(2019)
    # name = DataService.get_all_data(2020)
    startTime= time.time()

    X_train, X_test, y_train, y_test = AnalysisService.load_data_for_analysis(name)
    #AnalysisService.dataStats(data)
    AnalysisService.dt_analysis(X_train, X_test, y_train, y_test)
    AnalysisService.dt_analysis_pca(X_train, X_test, y_train, y_test)
    AnalysisService.logistic_regression_analysis(X_train, X_test, y_train, y_test)
    AnalysisService.logistic_regression_analysis_pca(X_train, X_test, y_train, y_test)
    #Do same analysis after using SMOTE
    X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled = AnalysisService.Smote(X_train, X_test, y_train, y_test)

    # all analysis below only use an oversampled trainingset
    AnalysisService.dt_analysis(X_train_oversampled, X_test, y_train_oversampled, y_test, SMOTER=True)
    AnalysisService.dt_analysis_pca(X_train_oversampled, X_test, y_train_oversampled, y_test, SMOTER=True)
    AnalysisService.logistic_regression_analysis(X_train_oversampled, X_test, y_train_oversampled, y_test, SMOTER=True)
    AnalysisService.logistic_regression_analysis_pca(X_train_oversampled, X_test, y_train_oversampled, y_test, SMOTER=True)
    #
    # # all analysis below use both oversampled training and test set
    AnalysisService.dt_analysis(X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled, SMOTER=True)
    AnalysisService.dt_analysis_pca(X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled, SMOTER=True)
    AnalysisService.logistic_regression_analysis(X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled, SMOTER=True)
    AnalysisService.logistic_regression_analysis_pca(X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled, SMOTER=True)
    #
    # nn_time = time.time()
    NeuralNetworkService.neural_network_analysis(X_train, X_test, y_train, y_test, "nn_result/nn")
    NeuralNetworkService.neural_network_analysis(X_train_oversampled, X_test, y_train_oversampled, y_test, "nn_result/nn_train_oversampled")
    NeuralNetworkService.neural_network_analysis(X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled, "nn_result/nn_oversampled")
    #print("neural network time = " + str(time.time() - nn_time))

    print("total time = " + str(time.time() - startTime))

    X, y = AnalysisService.load_data_for_analysis_speeding_score(name)
    oversampled_x, oversampled_y = AnalysisService.Smote(X, y)
    AnalysisService.speeding_score_analysis(oversampled_x, y)



   # data = AnalysisService.pca_analysis(data)
    #AnalysisService.logistic_regression_analysis(data)



   # AnalysisService.factor_analysis_mixed_data(data)


    # start = dt.datetime.now()
    # CephService.get_data_from_ceph("2020-07-01 00:00:00", "2020-07-31 23:59:59")
    # print(dt.datetime.now() - start)