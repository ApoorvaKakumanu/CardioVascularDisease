from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("cardio_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(x) for x in request.form.values()]
        input_data = np.array([values])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]

        if prediction >= 0.5:
            result = "High risk of cardiovascular disease"
        else:
            result = "Low risk of cardiovascular disease"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)