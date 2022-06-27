from flask import Flask,request,render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
df=pd.read_csv('SUV_Purchase.csv')

app=Flask(__name__)

#deSerialize
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[int(x) for x in request.form.values()]
    print(features)
    final=[np.array(features)]
    x=df.iloc[:,2:4].values
    sst=StandardScaler().fit(x)
    output=model.predict(sst.transform(final))
    print(output)
    if output[0]==0:
        return render_template('index.html',pred=f'The Person will not be able to buy the SUV Car')
    else:
        return render_template('index.html', pred=f'The Person will be able to buy the SUV Car')

if __name__=='__main__':
    app.run(debug=True)
