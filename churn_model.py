import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("Churn_Modelling.csv")
data.columns = data.columns.str.strip()

# Convert CustomerId to string
data['CustomerId'] = data['CustomerId'].astype(str).str.strip()

# Features for churn prediction
feature_columns = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'IsActiveMember',
    'EstimatedSalary'
]

X = data[feature_columns]
y = data['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# Get customer details
def get_customer_details(customer_id):
    customer = data[data['CustomerId'] == str(customer_id)]
    if customer.empty:
        return None
    return customer.iloc[0]


# Predict churn
def predict_churn(features):
    prediction = model.predict([features])
    probability = model.predict_proba([features])

    churn_prob = probability[0][1] * 100
    stay_prob = probability[0][0] * 100

    return prediction[0], round(stay_prob, 2), round(churn_prob, 2)


# Loan repayment probability (engineered feature)
def calculate_loan_repayment_probability(customer):

    credit = customer['CreditScore']
    salary = customer['EstimatedSalary']
    balance = customer['Balance']
    tenure = customer['Tenure']
    active = customer['IsActiveMember']

    score = 0

    score += (credit / 1000) * 40
    score += (salary / 200000) * 25
    score += (balance / 250000) * 15
    score += (tenure / 10) * 10
    score += (active * 10)

    repayment_probability = min(score, 100)
    default_probability = 100 - repayment_probability

    return round(repayment_probability, 2), round(default_probability, 2)