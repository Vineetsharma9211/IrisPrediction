from flask import Flask, request, render_template
import numpy as np
import pickle

with open("model/logistic_regression_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("model/label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from the form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Prepare input for prediction
        user_input = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Predict species
        prediction = classifier.predict(user_input_scaled)
        species_name = label_encoder.inverse_transform(prediction)

        return render_template("index.html", result=f"Predicted Species: {species_name[0]}")
    except Exception as e:
        return render_template("index.html", error="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
