import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, recall_score
import yaml
import os
import mlflow
from urllib.parse import urlparse # to get remote repository

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/sachsinghh/Machine-Learning-Pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']='sachsinghh'
os.environ['MLFLOW_TRACKING_PASSWORD']='-'

params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"], axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Load the model
    model = pickle.load(open(model_path, 'rb'))

    pred = model.predict(X)
    accuracy = accuracy_score(y, pred)
    recall = recall_score(y,pred)

    # log metrics
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("recall", recall)

    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')


if __name__ == '__main__':
    evaluate(params['data'], params['model'])

