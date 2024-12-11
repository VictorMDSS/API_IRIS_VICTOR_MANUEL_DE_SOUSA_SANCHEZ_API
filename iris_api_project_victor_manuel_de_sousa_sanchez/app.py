import os
from sklearn.metrics import accuracy_score

from flask import Flask, jsonify, request, json
from utils.iris_model import load_model, predict_species, load_data, train_model, save_model, download_data, load_or_initialize_model

#Configuracion de la aplicacion Flask

app= Flask(__name__)
app.config["DEBUG"] = True

#Ruta de los archivos del proyecto
DATA_PATH = "data/iris.csv"
MODEL_PATH = "models/iris_model.pkl"
CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}

#Descargar datos y cargar o inicializar el modelo
download_data(DATA_PATH)
model = load_or_initialize_model(DATA_PATH, MODEL_PATH)

#Ruta de bienvenida
@app.route("/", methods=["GET"])
def home():
    response = {
        "message": "Welcome to the Iris model prediction API",
        "endpoints": {
            '/api/v1/predict': 'Provides predictions based on input features (GET)',
            '/api/v1/retrain': 'Retrains the model with a new dataset (GET)',
            '/api/v1/accuracy': 'Shows the current accuracy of the model (GET)'},
        "example": {
            '/api/v1/predict?sepal_length=5.0&sepal_width=3.6&petal_length=1.4&petal_width=0.2': 
                'Add an endpoint like this one to predict the species of the Iris flower using the provided feature values'}
    }
    return app.response_class(
        response = json.dumps(response, ensure_ascii = False, indent = 4),
        status = 200,
        mimetype="application/json"
    )

#Realizar prediccion
@app.route("/api/v1/predict", methods = ["GET"])
def predict():
    try:
        sepal_length = float(request.args.get("sepal_length"))
        sepal_width = float(request.args.get("sepal_width"))
        petal_length = float(request.args.get("petal_length"))
        petal_width = float(request.args.get("petal_width"))
    except (TypeError, ValueError):
        return jsonify({"error": 
            "Please provide valid numeric values for all parameters: sepal_length, sepal_width, petal_length, petal_width"}), 400
    
    try: 
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        class_name = CLASS_MAPPING[int(prediction[0])]
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500
    
    return jsonify({
        "predintion": class_name,
        "input":{
            "sepal_length" : sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
    })
    
#Retrain the model with existing dataset and evaluate the new accuracy
@app.route("/api/v1/retrain", methods = ["GET"])
def retrain():
    if os.path.exists(DATA_PATH):
        try:
            X_train, X_test, y_train, y_test = load_data(data_path=DATA_PATH)
            model = train_model(X_train=X_train, y_train=y_train)
            save_model(model=model, model_path=MODEL_PATH)
            
            accuracy = accuracy_score(y_true = y_test, y_pred=model.predict(X_test))
            return jsonify({"message": "Model retrained succesfully", "accuracy": str(accuracy)})
        except Exception as e: 
            return jsonify({"error": f"An error occurred during retraining: {str(e)}"}), 500
        
    else: 
        return jsonify({"error": "Dataset for retraining not found"}), 404
    

#Calculate and return the accuracy of the current saved model on a validation dataset
@app.route("api/v1/accuracy", methods= ["GET"])
def accuracy():
    try:
        X_train, X_test, y_train, y_test = load_data(data_path=DATA_PATH)
        model = load_model(model_path=MODEL_PATH)
        accuracy = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
        return jsonify({"accuracy": str(accuracy)})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred during retraining: {str(e)}"}), 500


#Webhook
"""@app.route("/webhook", methods=["POST"])
def webhook():
    repo_path = "/home/vmdss/api_iris_model"
    server_wsgi = "/var/www/vmdss_pythonanywhere_com_wsgi.py"
    
    if request.is_json:
        subprocess.run(["git", "-C", repo_path, "pull"], check= True)
        subprocess.run(["touch", server_wsgi], check=True)
        return jsonify({"message": "Deployment update succesfully"}), 200
    else:
        return jsonify({"error": "Invalid request"}), 400
    """

if __name__ == "__main__":
    app.run()

    