
#from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import streamlit as st
#app = Flask(__name__)
#import flasgger
#from flasgger import Swagger
#Swagger(app)

pickle_in=open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)


#@app.route('/')
def welcome():
    return "Welcome"

#@app.route('/predict')
def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    """Let's authenticate the bank note
    This is the docstring for specifications.
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: the output values 
        
    """
    #variance=request.args.get('variance')
    #skewness=request.args.get('skewness')
    #curtosis=request.args.get('curtosis')
    #entropy=request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "the predicted value is "+str(prediction)


def  main():
    st.title("Bank authenticator")
    html_temp = """
    <div style:"background-color=red;padding:10px">
    <h1 style="text-align:center;color:white">Stream lit Bank Authenticator ML APP</h1>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input('variance',"type here")
    skewness = st.text_input('skewness',"type here")
    curtosis = st.text_input('curtosis',"type here")
    entropy = st.text_input('entropy',"type here")
    result=""
    if st.button("Predict"):
        result = predict_note_authentication(variance, skewness, curtosis, entropy)
    st.success('the output is {}'.format(result))
    if st.button("About"):
        st.text('lets Learn' )
        st.text('Built with  Streamlit')



if __name__ == "__main__":
    main()
    #app.debug = True
    #app.run()
