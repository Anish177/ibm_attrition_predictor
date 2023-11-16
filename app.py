from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

app = Flask(__name__)

# Joblib files obtained using these lines
# joblib.dump(logreg, 'ibm_attrition_predictor_logreg')
# joblib.dump(label_encoder, 'label_encoder.joblib')
# joblib.dump(scaler, 'scaler.joblib')

# Load the logistic regression model
logreg_model = joblib.load(r"ibm_attrition_predictor_logreg.joblib")

# Load label encoder and scaler
label_encoder = joblib.load(r"label_encoder.joblib")
scaler = joblib.load(r"scaler.joblib")

# Load the original dataset
# Assuming your original dataset is in a CSV file named 'original_dataset.csv'
# original_dataset = pd.read_csv(r"dataset.csv")  # Make sure to provide the correct file path

# Fit label encoder on all possible values of categorical columns
# for column in original_dataset.select_dtypes(include=['object']).columns:
#     label_encoder.fit(original_dataset[column])

# Define feature names
features = ["Age", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField",
            "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobRole",
            "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
            "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
            "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
            "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

# Define default values based on the provided data
# default_values = ["41", "1102", "Sales", "1", "2", "Life Sciences", "2", "Female", "94", "3", "Sales Executive",
#                   "4", "Single", "5993", "19479.0", "8", "Yes", "11", "3", "1", "0", "8", "0", "1", "6", "4", "0", "5"]

default_values = ["38", "423", "Human Resources", "11", "3", "Other", "2", "Male", "45", "4", "Manufacturing Director",
                  "3", "Divorced", "8407", "7842.0", "1", "No", "12", "4", "3", "1", "31", "6", "3", "13", "1", "10", "11"]


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        # Get user input from the form
        user_input = [request.form.get(feature) for feature in features]
        # label_encoder = LabelEncoder()
        # for column in X.select_dtypes(include=['object']).columns:
        #     X[column] = label_encoder.fit_transform(X[column])
        # Encode categorical variables using Label Encoding
        for i in [2, 5, 7, 10, 12, 16]:  # Indices of categorical columns in 'features'
            user_input[i] = label_encoder.fit_transform([user_input[i]])[0]

        # Convert features to float and scale numerical features
        user_input = np.array(user_input, dtype=float)
        user_input_scaled = scaler.transform([user_input])

        # Make a prediction
        prediction = logreg_model.predict(user_input_scaled)[0]

    return render_template('index.html', features=features, default_values=default_values, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
