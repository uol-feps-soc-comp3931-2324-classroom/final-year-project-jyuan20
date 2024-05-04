#References: https://rashik004.medium.com/using-lime-to-explain-ml-models-for-numeric-data-caad948c9503
#            https://www.freecodecamp.org/news/interpret-black-box-model-using-lime/
#            https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
#            https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
#            https://www.freecodecamp.org/news/interpret-black-box-model-using-lime/
#            https://towardsdatascience.com/complete-guide-on-model-deployment-with-flask-and-heroku-98c87554a6b9
#            https://www.analyticsvidhya.com/blog/2024/03/how-to-deploy-a-machine-learning-model-using-flask/

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from lime import lime_tabular
import re
import shap

# Initialize the Flask app
app = Flask(__name__)

# Load the model and encoder
model = pickle.load(open("models/xgb_model.sav", "rb"))
encoder = pickle.load(open("models/encoder_file.sav", "rb"))
x_train = pickle.load(open("models/x_train.sav", "rb"))
y_train = pickle.load(open("models/y_train.sav", "rb"))
shap_explainer = shap.TreeExplainer(model)

def get_age_group(age):
    if age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return '70-79'

def get_income_group(income):
    if income < 25000:
        return 'low'
    elif income < 50000:
        return 'low-middle'
    elif income < 75000:
        return 'middle'
    elif income < 100000:
        return 'high-middle'
    else:
        return 'high'

# Generate a list of feature names based on input data including categorical features.
def get_feature_names(input_data):
    feature_names = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'cb_person_default_on_file', 'cb_person_cred_hist_length',
        'loan_to_income_ratio', 'cb_person_default_on_file_encoded'
    ]

    home_ownership = f"person_home_ownership_{input_data['person_home_ownership']}"
    loan_intent = f"loan_intent_{input_data['loan_intent']}"
    feature_names.append(home_ownership)
    feature_names.append(loan_intent)

    return feature_names

# Extract the base feature name by removing conditions and numerical thresholds
def extract_base_feature_name(feature_with_condition):
    base_feature = re.split(' <= | >= | > | < ', feature_with_condition)[0]
    return base_feature.strip()

# Generate a user-friendly description of the LIME explanation
def generate_lime_explanation(lime_exp, feature_descriptions, features_to_explain):
    lime_description = {}
    for feature, effect in lime_exp:
        base_feature = extract_base_feature_name(feature)
        if base_feature in features_to_explain:
            user_friendly_feature = feature_descriptions.get(base_feature, base_feature)  
            impact_description = f"{user_friendly_feature} {'increases' if effect > 0 else 'decreases'} the risk by {abs(effect)*100:.2f}%"
            lime_description[user_friendly_feature] = impact_description
    return lime_description

#Format SHAP values into user-friendly explanations for each feature.
def format_shap_output(shap_values, feature_names, feature_descriptions):
    formatted_shap = []
    for name, value in zip(feature_names, shap_values[0]):  # Assuming binary classification
        description = feature_descriptions.get(name, name)
        impact = "increases" if value > 0 else "decreases"
        formatted_shap.append(f"{description} {impact} the risk by {abs(value):.2f} units.")
    return formatted_shap

@app.route('/', methods=["GET", "POST"])
def main():
    error = None  # Error handling variable
    lime_description = {}
    if request.method == "POST":
        try:
            # Extract and preprocess form data
            person_age = float(request.form.get('person_age', 0))
            person_income = float(request.form.get('person_income', 0))
            person_emp_length = float(request.form.get('person_emp_length', 0))
            loan_amnt = float(request.form.get('loan_amnt', 0))
            loan_int_rate = float(request.form.get('loan_int_rate', 0))
            cb_person_cred_hist_length = float(request.form.get('cb_person_cred_hist_length', 0))
            
            # Compute additional features based on inputs
            age_group = get_age_group(person_age)
            income_group = get_income_group(person_income)
            loan_to_income_ratio = loan_amnt / person_income if person_income > 0 else 0

            # Encode categorical data
            categorical_data = pd.DataFrame([[
                request.form['person_home_ownership'],
                request.form['loan_intent'],
                age_group,
                income_group
            ]], columns=['person_home_ownership', 'loan_intent', 'age_group', 'income_group'])
            categorical_encoded = encoder.transform(categorical_data).toarray()
            
            # Combine all features for model prediction
            numerical_data = np.array([[person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, cb_person_cred_hist_length, loan_to_income_ratio]])
            default_encoded = np.array([[int(request.form.get('cb_person_default_on_file', 'N') == 'Y')]])
            features = np.hstack([numerical_data, categorical_encoded, default_encoded])
            feature_names = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                            'loan_int_rate', 'cb_person_cred_hist_length', 'loan_to_income_ratio',
                            'cb_person_default_on_file_encoded', 'person_home_ownership_MORTGAGE',
                            'person_home_ownership_OTHER', 'person_home_ownership_OWN',
                            'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
                            'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
                            'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                            'age_group_20-29', 'age_group_30-39', 'age_group_40-49',
                            'age_group_50-59', 'age_group_60-69', 'age_group_70-79',
                            'income_group_high', 'income_group_high-middle', 'income_group_low',
                            'income_group_low-middle', 'income_group_middle']
            feature_descriptions = {
                'person_age': 'Age',
                'person_income': 'Income',
                'person_emp_length': 'Employment Length (in years)',
                'loan_amnt': 'Loan Amount',
                'loan_int_rate': 'Loan Interest Rate',
                'cb_person_cred_hist_length': 'Length of Credit History',
                'loan_to_income_ratio': 'Ratio of Loan Amount to Income',
                'cb_person_default_on_file_encoded': 'History of Default',
                'person_home_ownership_MORTGAGE': 'Homw Ownership: Mortgage',
                'person_home_ownership_OTHER': 'Home Ownership: Other',
                'person_home_ownership_OWN': 'Home Ownership: Own',
                'person_home_ownership_RENT': 'Home Ownership: Rent',
                'loan_intent_DEBTCONSOLIDATION': 'Loan Intent: Debt Consolidation',
                'loan_intent_EDUCATION': 'Loan Intent: Education',
                'loan_intent_HOMEIMPROVEMENT': 'Loan Intent: Home Improvement',
                'loan_intent_MEDICAL': 'Loan Intent: Medical',
                'loan_intent_PERSONAL': 'Loan Intent: Personal',
                'loan_intent_VENTURE': 'Loan Intent: Venture'  
            }
            # Predict risk using model
            prediction = model.predict(features)[0]
            risk_label = 'High Risk' if prediction == 1 else 'Low Risk'  

            # Explain prediction using LIME and SHAP
            features_to_explain = get_feature_names(request.form)
            explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(x_train),
            feature_names=feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
            )
            lime_exp = explainer.explain_instance(data_row=features[0], predict_fn=model.predict_proba, num_features=30)
            lime_description = generate_lime_explanation(
                lime_exp.as_list(), 
                feature_descriptions, 
                features_to_explain
            )
            shap_values = shap_explainer.shap_values(features)  
            formatted_shap_explanations = format_shap_output(shap_values, features_to_explain, feature_descriptions)

            #Render the results
            return render_template("index.html", result=risk_label, lime=lime_description, shap_explanations=formatted_shap_explanations, original_input=request.form.to_dict())
        except Exception as e:
            error = str(e)
            print(f"Error: {error}")
            return render_template("index.html", error=error, original_input=request.form.to_dict())

    return render_template("index.html", original_input={}, result=None)

if __name__ == '__main__':
    app.run(debug=True)