from flask import Flask, render_template, request
from churn_model import (
    predict_churn,
    get_customer_details,
    calculate_loan_repayment_probability
)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    customer_id = request.form.get('customer_id', '').strip()

    if not customer_id:
        return render_template(
            'index.html',
            prediction_text="Please enter a Customer ID."
        )

    customer = get_customer_details(customer_id)

    if customer is None:
        return render_template(
            'index.html',
            prediction_text="Customer ID not found in dataset."
        )

    customer_dict = customer.to_dict()

    # Extract features
    features = [
        float(customer_dict['CreditScore']),
        float(customer_dict['Age']),
        float(customer_dict['Tenure']),
        float(customer_dict['Balance']),
        float(customer_dict['NumOfProducts']),
        float(customer_dict['IsActiveMember']),
        float(customer_dict['EstimatedSalary'])
    ]

    # Churn prediction
    prediction, stay_prob, churn_prob = predict_churn(features)

    # Loan repayment
    repay_prob, default_prob = calculate_loan_repayment_probability(customer_dict)

    if prediction == 1:
        result = f"Customer {customer_id} is likely to churn."
    else:
        result = f"Customer {customer_id} is likely to stay."

    return render_template(
        'index.html',
        prediction_text=result,
        stay=stay_prob,
        churn=churn_prob,
        repay=repay_prob,
        default=default_prob,
        customer=customer_dict
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)