from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score
from sklearn.metrics  import accuracy_score

class ModelFinder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: Muhammad Asad Majeed
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = GradientBoostingClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def Get_Best_Params_For_Gradient_Boosting_Classifier(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_Gradient_Boosting_Classifier
                                Description: get the parameters for Gradient_Boosting_Classifier Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Muhammad Asad Majeed
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_Gradient_Boosting_Classifier method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {                
                                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                                'max_depth': [3, 5, 10, 20],
                                'n_estimators': [10, 50, 100, 200]
                                }

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'GBDT best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_GBDT method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_GBDT method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'GBDT Parameter tuning  failed. Exited the get_best_params_for_GBDT method of the Model_Finder class')
            raise Exception()

    def Get_Best_Params_For_Xgboost(self, train_x, train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Muhammad Asad Majeed
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' 
                                   + str(self.grid.best_params_) 
                                   + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' 
                                   + str(e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def Get_Best_Model(self, train_x, train_y, test_x, test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Muhammad Asad Majeed
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost = self.Get_Best_Params_For_Xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for GBDT
            self.gbdt = self.Get_Best_Params_For_Gradient_Boosting_Classifier(train_x,train_y)
            self.prediction_gbdt=self.gbdt.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.gbdt_score = accuracy_score(test_y,self.prediction_gbdt)
                self.logger_object.log(self.file_object, 'Accuracy for GBDT:' + str(self.gbdt_score))
            else:
                self.gbdt_score = roc_auc_score(test_y, self.prediction_gbdt) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for GBDT:' + str(self.gbdt_score))

            #comparing the two models
            if(self.gbdt_score <  self.xgboost_score):
                return 'XGBoost', self.xgboost
            else:
                return 'GBDTClassifier', self.gbdt

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' 
                                   + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

