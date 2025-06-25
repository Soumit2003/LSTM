from flask import Flask, request, render_template_string
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

linear_model_path =r"C:\Users\KIIT\Desktop\AD_LAB\exp3\linear_regression_model.joblib"
scaler_X_path = r"C:\Users\KIIT\Desktop\AD_LAB\exp3\scaler_X.pkl"
lstm_model_path = r"C:\Users\KIIT\Desktop\AD_LAB\exp3\lstm_model (2).h5"
lr_model, lstm_model, scaler_X = None, None, None

try:
    lr_model = joblib.load(linear_model_path)
    print("Linear Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading Linear Regression model: {e}")

try:
    lstm_model = load_model(lstm_model_path)
    print("LSTM model loaded successfully.")
except Exception as e:
    print(f"Error loading LSTM model: {e}")

try:
    scaler_X = joblib.load(scaler_X_path)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            color: #333;
        }
        form {
            max-width: 400px;
            margin: auto;
        }
        input[type="text"], select, button {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>Prediction App</h1>
    <form method="POST" action="/predict">
        <label for="model">Select Model:</label>
        <select name="model" required>
            <option value="linear">Linear Regression</option>
            <option value="lstm">LSTM</option>
        </select>
        <label for="feature1">Feature 1:</label>
        <input type="text" name="feature1" required>
        <label for="feature2">Feature 2:</label>
        <input type="text" name="feature2" required>
        <label for="feature3">Feature 3:</label>
        <input type="text" name="feature3" required>
        <label for="feature4">Feature 4:</label>
        <input type="text" name="feature4" required>
        <label for="feature5">Feature 5:</label>
        <input type="text" name="feature5" required>
        <label for="feature6">Feature 6:</label>
        <input type="text" name="feature6" required>
        <label for="feature7">Feature 7:</label>
        <input type="text" name="feature7" required>
        <label for="feature8">Feature 8:</label>
        <input type="text" name="feature8" required>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
    <div class="result">
        <strong>Predicted Value:</strong> {{ prediction }}
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('model')  # Selected model
        features = [float(request.form[f'feature{i}']) for i in range(1, 9)]
        input_data = np.array([features])

        if scaler_X:
            scaled_input = scaler_X.transform(input_data)
        else:
            return render_template_string(html_template, prediction="Error: Scaler not loaded.")

        if model_type == "linear" and lr_model:
            prediction = lr_model.predict(scaled_input)
        elif model_type == "lstm" and lstm_model:
            lstm_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))  # Reshape for LSTM
            prediction = lstm_model.predict(lstm_input)
        else:
            return render_template_string(html_template, prediction="Error: Model not available.")

        return render_template_string(html_template, prediction=round(prediction[0][0], 2))
    except Exception as e:
        return render_template_string(html_template, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
