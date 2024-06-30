from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Sample database of valid values
valid_areas = [
    'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
    'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria',
    'Burkina Faso', 'Burundi', 'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia',
    'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 'Estonia',
    'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
    'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya',
    'Latvia', 'Lebanon', 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali',
    'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
    'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway', 'Pakistan', 'Papua New Guinea',
    'Peru', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia',
    'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan',
    'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe'
]
valid_items = ['Maize', 'Potatoes', 'Rice, paddy','Sorghum', 'Soyabeaans', 'Wheat','Cassava', 'Sweet potatoes', 'Plantains and others','Yarns']

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Initialize error messages
        error_messages = []

        # Check if any field is empty
        if not all([Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]):
            error_messages.append("Please fill in all fields.")

        # Validate numeric inputs
        try:
            if Year:
                Year = int(Year)
            if average_rain_fall_mm_per_year:
                average_rain_fall_mm_per_year = float(average_rain_fall_mm_per_year)
            if pesticides_tonnes:
                pesticides_tonnes = float(pesticides_tonnes)
            if avg_temp:
                avg_temp = float(avg_temp)
        except ValueError:
            error_messages.append("Please enter valid numeric values for numerical fields.")

        # Validate Area
        if Area and Area not in valid_areas:
            error_messages.append(f"Entered Area '{Area}' is not valid. Please enter a valid area like: {', '.join(valid_areas)}")

        # Validate Item
        if Item and Item not in valid_items:
            error_messages.append(f"Entered Item '{Item}' is not valid. Please enter a valid item like: {', '.join(valid_items)}")

        # If there are any errors, render template with error messages
        if error_messages:
            return render_template('index.html', error_messages=error_messages)

        # Create features array and transform using preprocessor
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)

        # Predict using the model
        prediction = dtr.predict(transformed_features)
        
        # Depending on your model output, you might need to reshape or format the prediction
        # For example, if prediction is multidimensional, ensure it's in a suitable format
        prediction = prediction[0]  # Assuming prediction is a single value

        return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
