from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("idfy.html")


@app.route('/predict', methods=['POST','GET'])
def predict():
    input = []
    age = request.form.get('age')
    input.append(age)
    WorkClass = request.form.get('WorkClass')
    
    fnlwgt = request.form.get('fnlwgt')
    input.append(fnlwgt)
    Education = request.form.get('Education')
    
    ENY = request.form.get('ENY')
    input.append(ENY)
    Marital_Status = request.form.get('Marital-Status')
    input.append(Marital_Status)
    Occupation = request.form.get('Occupation')
    
    Relationship = request.form.get('Relationship')
    
    Race = request.form.get('Race')
    
    Sex = request.form.get('Sex')
    input.append(Sex)
    Capital_Gain = request.form.get('Capital-Gain')
    input.append(Capital_Gain)
    Capital_loss = request.form.get('Capital-loss')
    input.append(Capital_loss)
    Hours_per_week = request.form.get('Hours-per-week')
    input.append(Hours_per_week)
    Native_Country= request.form.get('Native-Country')
  
    income= request.form.get('income')


    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
     'marital.status', 'occupation', 'relationship', 'race', 'sex',
      'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']

    input[4] = input[4].replace("Male","1" )
    input[4] = input[4].replace("Female","0")
    input[4]= int(input[4])

    
     

   

    
    input[3] = input[3].replace('Never-married', 'Single')
    input[3] = input[3].replace('Divorced', 'Single')
    input[3]= input[3].replace('Separated', 'Single')
    input[3]= input[3].replace('Widowed', 'Single')
    input[3] = input[3].replace('Married-civ-spouse', 'Married')
    input[3]= input[3].replace('Married-spouse-absent', 'Married')
    input[3]= input[3].replace('Married-AF-spouse', 'Married')

    input[3] = input[3].replace("Married","1" )
    input[3] = input[3].replace("Single","0")
    input[3]= int(input[3])
    input = np.array(input)
    print(input)
    # Drop the data you don't want to use
    np.reshape(input,(1,-1))
    print(input)
    
    output = model.predict(input)

    if output:
        return render_template('forest_fire.html',pred='Thank you for your details')
    else:
       return render_template('forest_fire.html',pred='PLease enter valid details')


if __name__ == '__main__':
    app.run(debug=True)
