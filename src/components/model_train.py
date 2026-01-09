from src.exception.exception import CustomerException
from src.logging.logging import logging

from src.entity.artifacts_entity.artifacat_entity import Data_transformation_artifact,Model_train_artifact
from src.entity.config_entity.config_entity import Model_Train_Config
from src.utils.utils import save_object,load_object

from src.utils.model_utils import perform_hyperparameter_tuning


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
                "Random Forest": RandomForestClassifier(class_weight="balanced"),
                "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
                "Logistic Regression": LogisticRegression(class_weight="balanced",max_iter=1000),
                "Naive Bayes": GaussianNB(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XG Boost": XGBClassifier(),
                "SVC": SVC()

            }

            param_grids = {

                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"]
                },

                "Random Forest": {
                    "n_estimators": [100, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },

                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 11],
                    "weights": ["uniform", "distance"]
                },

                "SVC": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1]
                },

                "XG Boost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "random_state":42,
                    "use_label_encoder":False,
                    "eval_metric":"logloss",
                }
            }

            model_scores=[]

            for model_name,model in models.items():

                logging.info(f"Training Model: {model_name}")

                model.fit(X_train,y_train)

                train_acc,train_f1_score = self.evaluate_model(model,X_train,y_train)
                test_acc,test_f1_score = self.evaluate_model(model,X_test,y_test)


                model_scores.append({"model_name":model_name,
                                     "model":model,
                                     "train_f1":train_f1_score,
                                     "test_f1":test_f1_score})
                
                logging.info(model_scores)

            ## select top two models
            model_scores = sorted(model_scores,key=lambda x:x['test_f1'],reverse=True)

            top_models = model_scores[:2]

            best_model = None
            best_test_f1 = -1    
            best_model_name = None
            best_train_f1 = None

            for mode_info in top_models:
                model_name = mode_info["model_name"]
                model = mode_info["model"]

                if model_name not in param_grids:
                    logging.info(f"Skipping hyperparameter tuning for {model_name}")
                    continue

                logging.info(f"Tuning model: {model_name}")

                tuned_model, best_params, best_cv_f1 = perform_hyperparameter_tuning(
                    model=model,
                    pram_grid=param_grids[model_name],
                    X_train=X_train,
                    y_train=y_train
                )

                logging.info(
                    f"{model_name} | Best Params: {best_params} | "
                    f"CV F1: {best_cv_f1:.4f}"
                )

                train_acc, train_f1 = self.evaluate_model(
                    tuned_model, X_train, y_train
                )
                test_acc, test_f1 = self.evaluate_model(
                    tuned_model, X_test, y_test
                )

                logging.info(
                    f"{model_name} | "
                    f"Tuned Train F1: {train_f1:.4f}, "
                    f"Tuned Test F1: {test_f1:.4f}"
                )

                if test_f1 > best_test_f1:
                    best_model = tuned_model
                    best_model_name = model_name
                    best_test_f1 = test_f1
                    best_train_f1 = train_f1



            if best_model is None:
                    raise CustomerException("No Model Performed Well...",sys)
            


            if best_test_f1 < self.model_train_config.threshold_accuracy:
                raise CustomerException(f"Best model accuracy {best_test_f1} is below expected threshold",sys)

            save_object(self.model_train_config.model_train_file_path,best_model)

            save_object("final_pikels/model.pkl",best_model)
            
            logging.info(f"Best Model Saved: {best_model_name} ")   
            
            return best_model,best_model_name,best_train_f1,best_test_f1



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