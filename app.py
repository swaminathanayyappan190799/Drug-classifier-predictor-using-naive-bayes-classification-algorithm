import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('Drugpredict.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method=='POST':
		age=int(request.form["age"])
		gen=request.form["gender"]
		bp=request.form["bplev"]
		chol=request.form["chollev"]
		sodtopot=int(request.form["sodtopot"])
		temp1=0.0
		temp2=0.0
		gen1=0
		ch=0
		if bp=="HIGH":
  			temp1=0.0
  			temp2=0.0
		elif bp=="LOW":
  			temp1=1.0
  			temp2=0.0
		elif bp=="NORMAL":
  			temp1=0.0
  			temp2=1.0
		if gen=="M":
  			gen1=1
		elif gen=="F":
  			gen1=0
		if chol=="HIGH":
  			ch=0
		elif chol=="NORMAL":
  			ch=1
		result=int(model.predict([[temp1,temp2,age,gen1,ch,sodtopot]]))
		if result==0:
			return render_template('Drugpredict.html',prediction_text="The recommended drug is : DrugY")
		elif result==1:
			return render_template('Drugpredict.html',prediction_text="The recommended drug is : drugA")
		elif result==2:
			return render_template('Drugpredict.html',prediction_text="The recommended drug is : drugB")
		elif result==3:
			return render_template('Drugpredict.html',prediction_text="The recommended drug is : drugC")
		elif result==4:
			return render_template('Drugpredict.html',prediction_text="The recommended drug is : drugX")
        
  			



if __name__=='__main__':
    app.run(debug=True)