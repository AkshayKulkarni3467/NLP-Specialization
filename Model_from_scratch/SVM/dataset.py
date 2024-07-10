import pandas as pd
from sklearn.preprocessing import StandardScaler


def dataset(path):
    df = pd.read_csv(path)

    X = df.drop(columns=['Outcome','Pregnancies','DiabetesPedigreeFunction','SkinThickness'])
    Y = df['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)

    features = standardized_data
    targets = Y
    
    return features, targets