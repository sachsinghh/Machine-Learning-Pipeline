import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import pickle
import yaml
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse # to get remote repository
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/sachsinghh/Machine-Learning-Pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']='sachsinghh'
os.environ['MLFLOW_TRACKING_PASSWORD']='-'

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"], axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    #start mlflow run
    with mlflow.start_run():
        #split dataset
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        signature = infer_signature(X_train, y_train)

        #Hyperparam grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        #HYPERPARAM TUNING
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        #best model
        best_model = grid_search.best_estimator_

        #predict outcome and evaluate mdoel
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print(f'Accuracy:{accuracy}')
        print(f'Recall:{recall}')

        #log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("best_n_estimator", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        #log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file': # see if file is http or not
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, 'model', signature=signature)

        # create model directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True) # if path exists then dont do anything to the particular file
        filename = model_path
        pickle.dump(best_model,open(filename, 'wb'))

        print(f'Model Saved Successfully to {model_path}')


if __name__ == '__main__':
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])



