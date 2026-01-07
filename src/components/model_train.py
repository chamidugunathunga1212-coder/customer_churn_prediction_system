from src.exception.exception import CustomerException
from src.logging.logging import logging

from src.entity.artifacts_entity.artifacat_entity import Data_transformation_artifact,Model_train_artifact
from src.entity.config_entity.config_entity import Model_Train_Config
from src.utils.utils import save_object,load_object


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,f1_score


from src.utils.utils import load_numpy_array_data

import sys
import os


class ModelTrainer:

    def __init__(self,model_train_config:Model_Train_Config,data_transformation_artifact:Data_transformation_artifact):

        try:
            self.model_train_config = model_train_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomerException(e,sys)
        
    def evaluate_model(self,model,X,y):
        try:

            predictions = model.predict(X)
            acc_score = accuracy_score(y,predictions)
            f1_sco = f1_score(y,predictions)

            return acc_score,f1_sco

        except Exception as e:
            raise CustomerException(e,sys)    
        

    def train_model(self,X_train,y_train,X_test,y_test):
        try:

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(verbose=1),
                "Naive Bayes": GaussianNB(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "K Neighbors": KNeighborsClassifier(),
                "XG Boost": XGBClassifier(),
                "SVC": SVC(verbose=1)

            }

            best_model = None
            best_f1 = -1    
            best_model_name = None
            best_train_f1 = None

            for model_name,model in models.items():

                logging.info(f"Training Model: {model_name}")

                model.fit(X_train,y_train)

                train_acc,train_f1_score = self.evaluate_model(model,X_train,y_train)
                test_acc,test_f1_score = self.evaluate_model(model,X_test,y_test)

                logging.info(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

                logging.info(f"{model_name} ->Training F1 score: {train_f1_score}, Test F1 score: {test_f1_score}")

                if test_f1_score > best_f1:
                    best_model = model
                    best_f1 = test_f1_score
                    best_model_name = model_name
                    best_train_f1 = train_f1_score

            if best_model is None:
                    raise CustomerException("No Model Performed Well...",sys)

            if best_f1 < self.model_train_config.threshold_accuracy:
                raise CustomerException(f"Best model accuracy {best_f1} is below expected threshold",sys)

            save_object(self.model_train_config.model_train_file_path,best_model)
            logging.info(f"Best Model Saved: {best_model_name} ")   
            
            return best_model,best_model_name,best_train_f1,best_f1



        except Exception as e:
            raise CustomerException(e,sys)    
        


    def initiate_model_trainer(self)->Model_train_artifact:
        try:
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## load the numpy array

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)


            X_train,y_train,X_test,y_test = (

                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )

            best_model,best_model_name,best_train_f1,best_test_f1= self.train_model(X_train,y_train,X_test,y_test)


            return Model_train_artifact(
                trained_model_file_path=self.model_train_config.model_train_file_path,
                train_metric_artifact=best_train_f1,
                test_metric_artifact=best_test_f1,
                best_model_name=best_model_name
            )
            
            





        except Exception as e:
            raise CustomerException(e,sys)    