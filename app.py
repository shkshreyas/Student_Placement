from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        IQ = float(request.form['IQ'])
        CGPA = float(request.form['CGPA'])
        final_features = np.array([[IQ, CGPA]])
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'
    except Exception as e:
        output = f"Error in prediction: {str(e)}"
    
    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
