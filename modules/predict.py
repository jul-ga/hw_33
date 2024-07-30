# <YOUR_IMPORTS>
import os
import dill
import pandas as pd
import json


path = os.environ.get('PROJECT_PATH', '.')
path_to_models = f'{path}/data/models'
path_to_test = f'{path}/data/test'
path_to_pred = f'{path}/data/predictions'


def download_best_model(path_to_models):
    files = os.listdir(path_to_models)
    for file_name in files:
        file_path = f'{path_to_models}/{file_name}'
        with open(file_path, 'rb') as file:
            model = dill.load(file)
        return model


def download_test_data(path_to_test):
    data = []
    files = os.listdir(path_to_test)
    for file_name in files:
        file_path = f'{path_to_test}/{file_name}'
        with open(file_path, 'r') as file:
            file_data = json.load(file)
            data.append(file_data)
    return pd.DataFrame(data)


def predict():
    model = download_best_model(path_to_models)
    test_data = download_test_data(path_to_test)
    predictions = model.predict(test_data)
    predictions_df = pd.DataFrame({
        'id': test_data['id'],
        'predictions': predictions
    })
    predictions_df.to_csv(f'{path_to_pred}/predictions.csv', index=False)


if __name__ == '__main__':
    predict()
