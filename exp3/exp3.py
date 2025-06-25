from flask import Flask, request, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

model_path = r"C:\Users\KIIT\Desktop\AD_LAB\exp3\lstm_model (2).h5"
scaler_X_path = r"C:\Users\KIIT\Desktop\AD_LAB\exp3\scaler_X.pkl"
scaler_y_path = r"C:\Users\KIIT\Desktop\AD_LAB\exp3\scaler_y.pkl"

model, scaler_X, scaler_y = None, None, None

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    scaler_X = joblib.load(scaler_X_path)
    print("Scaler X loaded successfully.")
except Exception as e:
    print(f"Error loading scaler X: {e}")

try:
    scaler_y = joblib.load(scaler_y_path)
    print("Scaler Y loaded successfully.")
except Exception as e:
    print(f"Error loading scaler Y: {e}")

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
        input[type="text"], button {
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
    <h1>LSTM Model Prediction</h1>
    <form method="POST" action="/predict">
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
        if model is None or scaler_X is None or scaler_y is None:
            raise ValueError("Model or scalers are not loaded properly. Please check the paths.")
        
        features = []
        for i in range(1, 9):  # For feature1 to feature8
            feature_value = request.form[f'feature{i}']
            try:
                features.append(float(feature_value))
            except ValueError:
                return render_template_string(html_template, prediction="Error: All features must be numeric.")
        
        input_data = np.array([features])
        
        scaled_input = scaler_X.transform(input_data)
        
        lstm_input = np.reshape(scaled_input, (scaled_input.shape[0], scaled_input.shape[1], 1))
        
        prediction_scaled = model.predict(lstm_input)
        
    
        prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        return render_template_string(html_template, prediction=round(prediction, 2))
    
    except Exception as e:
        return render_template_string(html_template, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
