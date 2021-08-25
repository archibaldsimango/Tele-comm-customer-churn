from flask import Flask,render_template,request
import pickle
import pandas as pd

model1 = pickle.load(open('xgbmodel.pkl','rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["GET","POST"])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == "POST":
        SeniorCitizen= request.form['SeniorCitizen']
        Partner = request.form['Partner']
        Dependents = request.form['Dependents']
        tenure = request.form['tenure']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        Contract = request.form['Contract']  
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        MonthlyCharges = request.form['MonthlyCharges'] 
        TotalCharges = request.form['TotalCharges'] 

        input_variables = pd.DataFrame([[SeniorCitizen, Partner, Dependents,tenure, OnlineSecurity,
       OnlineBackup, DeviceProtection, TechSupport, Contract,
       PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]],columns=['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

        prediction1 = model1.predict(input_variables)
        prob1 = model1.predict_proba(input_variables)[0]
        factor1 = prob1[0] *100
        confidence_factor1 = str(round(factor1, 1)) +'%'
        if prediction1 == 0:
            result1 = 'YES'
        else:
            result1 = 'NO'

        return render_template('main.html',result1=result1,confidence_factor1=confidence_factor1)

    if __name__ == '__main__':
        app.debug = True
        app.run()
        