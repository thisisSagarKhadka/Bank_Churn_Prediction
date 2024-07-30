from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_xgboost.pkl')

# Initialize LabelEncoders
label_encoders = {}

# Function to apply label encoding to categorical columns
# def apply_label_encoding(df):
#     for col in ['Gender', 'Ethnicity', 'Jaundice', 'Autism', 'Country', 'Used_app_before', 'Age_desc', 'Relation']:
#         label_encoders[col] = LabelEncoder()
#         df[col] = label_encoders[col].fit_transform(df[col])
#     return df

# Define the route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',input_data={})
# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.form
    
    # Prepare input data
    input_data = {
        # "RowNumber": data['RowNumber'],
        # "CustomerId": data['CustomerId'],
        # "Surname": data['Surname'],
        "CreditScore": data['CreditScore'],
        "Geography": data['Geography'],
        "Gender": data['Gender'],
        "Age": data['Age'],
        "Tenure": data['Tenure'],
        "Balance": data['Balance'],
        "NumOfProducts": data['NumOfProducts'],
        "HasCrCard": data['HasCrCard'],
        "IsActiveMember": data['IsActiveMember'],
        "EstimatedSalary": data['EstimatedSalary'],
        # "AgeGroup":data['AgeGroup'],
        # "Exited": data['Exited']
    }
    
    # Convert input_data to a DataFrame
    input_data = pd.DataFrame([input_data])
    print(input_data)
    # Apply label encoding
    # input_data_df = apply_label_encoding(input_data_df)
    
    # Make predictions
    predictions = model.predict(input_data)

    if predictions == 1:
        predictions = "Yes"
    else:
        predictions = "No"
    
    # Convert predictions to a list and return as JSON
    # return jsonify(predictions.tolist())
   
    return render_template('index.html', prediction_text='The prediction is {}'.format(predictions),input_data=input_data)

if __name__== '__main__':
    app.run(debug=True)

    