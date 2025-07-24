from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Dict

def main() -> None:
    dataset = pd.read_csv('src/data/auto-mpg.csv')

    for col in dataset.columns:
        print(f'a coluna {col} tem type {dataset[col].dtype} e tem {dataset[col].isna().sum()} missing values')
        
    ## preprocessing
    mean = 0.0
    count = 0
    for row in range(len(dataset['horsepower'])):
        try:
            mean += float(dataset['horsepower'][row])
            count += 1
        except Exception:
            pass
    mean = mean/count
    dataset.loc[dataset['horsepower'].str.contains(r'\?', na=False), 'horsepower'] = mean
    dataset['horsepower'] = pd.to_numeric(dataset['horsepower'])
    del dataset['car name']

    Y = dataset.pop('mpg')
    X = dataset

    ## Training

    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    linear_model = LinearRegression().fit(x_scaled,y_train)
    y_pred = linear_model.predict(scaler.transform(x_test))

    print(mean_squared_error(y_test, y_pred))


    ## Serialize the model

    pickle.dump(linear_model, open('src/data/model.pkl','wb'))
    return None


if __name__ == "__main__":
    main()