from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open(r'model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved scaler

app = Flask(__name__)

# Route for the main HTML page
@app.route('/')
def home():
    return render_template('abc.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    ig = float(request.form['ig'])
    cgpa = float(request.form['cgpa'])

    # Prepare data for prediction
    input_data = np.array([[cgpa, ig]])

    # Scale input data using the pre-fitted scaler
    #input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data)

    # Convert prediction to readable text
    placement_status = 'Placed' if prediction == 1 else 'Not Placed'
    
    # Return the result as a JSON object
    return jsonify({'placement': placement_status})

if __name__ == '__main__':
    app.run(debug=True)
