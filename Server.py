from flask import Flask, request, jsonify
#import util
import pickle
import json
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods= ['GET','POST'])
def Get_Date_For_Prediction():
    Date = float(request.form['Date for estimation'])
    response = jsonify({
    'estimated_price': util.get_estimated_price()
    })
        
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def get_estimated_price():
    loaded_model = joblib.load("C:/Users/Youssef Abuelleil/Desktop/Intern/Server/artifacts/Python_model.sav")
    Input=eval(input("please write date format in year/month/day"))
    Input = float(Input)
    z= np.array([[Input]])
    y_pred = loaded_model.predict(z)
    print(y_pred)
    return loaded_model.predict(z)

#@app.route('/predict_home_price', methods=['GET', 'POST'])
#def predict_home_price():
#    total_sqft = float(request.form['total_sqft'])
#    response = jsonify({
#        'estimated_price': util.get_estimated_price()
#    })
#    response.headers.add('Access-Control-Allow-Origin', '*')



if __name__ == "__main__":
    print("Starting python flask server")
    app.run()