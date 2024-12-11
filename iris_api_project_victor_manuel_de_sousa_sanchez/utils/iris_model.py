import pandas as pd 
import numpy as np 
import joblib
import os 
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

DATA_PATH = "data/iris.csv"
MODEL_PATH = "models/iris_model.pkl"
CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}

#Cargar el dataset Iris
def download_data(data_path):
    """Descarga y guarda el dataset IRIS desde un repositorio o URL"""
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columnas = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(url, names=columnas)
    df.to_csv(data_path, index = False)

#Dividir en datos de entrenamiento y prueba 
def load_data(data_path = DATA_PATH):
    """Carga el dataset IRIS y divide en conjuntos de entrenamiento y validación"""
    df = pd.read_csv(data_path)
    X = df.drop("class", axis = 1)
    y = df["class"]
    
    #Division de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

#Crear y entrenar el modelo 
def load_or_initialize_model(data_path, model_path):
    """Carga el modelo si ya existe, o lo inicializa y entrena si no."""
    if os.path.exists(model_path):
        return(load_model(model_path))
    else:
        X_train, X_test, y_train, y_test = load_data(data_path)
        model = train_test_split(X_train, y_train)
        save_model(model,model_path)
        return model
    
def train_model(X_train, y_train):
    """Entrena un modelo de clasificacion usando RandomForestClassifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Guarda el modelo entrenado en un archivo pickle."""
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
        
def load_model(model_path):
    """Carga un modelo desde un archivo pickle."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def predict_species(model, features):
    """Realiza una predicción usando el modelo proporcionado."""
    prediction = model.predict([features])[0]
    return CLASS_MAPPING[prediction]
