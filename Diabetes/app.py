import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create Flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Define mapping between numerical predictions and labels
label_map = {0: "Non-Diabetic", 1: "Diabetic"}

@flask_app.route("/")
def Home():
    return render_template('diabetes.html')

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Convert form values to float features
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Predict using the loaded model
    prediction = model.predict(features)
    
    # Map numerical prediction to label
    prediction_label = label_map[prediction[0]]
    
    # Render template with prediction label
    return render_template('diabetes.html', prediction_text="The person is {}".format(prediction_label))

if __name__ == "__main__":
    flask_app.run(debug=True)
