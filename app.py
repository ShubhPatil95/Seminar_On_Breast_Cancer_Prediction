import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) 
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['texture_mean','area_mean','concavity_mean','area_se',
                     'concavity_se','fractal_dimension_se','smoothness_worst',
                     'concavity_worst','symmetry_worst','fractal_dimension_worst']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    
    print(df,output)
    if output == 0:
        res_val = "Breast Cancer"
    else:
        res_val = "No Breast Cancer"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
#     app.debug = True
    app.run(port=8080)