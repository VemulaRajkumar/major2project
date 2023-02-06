import streamlit as st
import pickle
import numpy as np

model=pickle.load(open('randomForestClassifier.pkl','rb'))
crops=pickle.load(open('cropsDict.pkl','rb'))

def main():
    st.title("CROP PREDICTION USING ML")

    n=st.slider('Give Nitrogen percentage')
    p=st.slider('Give Phosphorous percentage')
    k=st.slider('Give Potassium percentage')
    temp = st.slider('Enter temperature : ',min_value=12,max_value=30,step=1)
    humidity = st.slider('Enter humidity : ',min_value=1,max_value=100,step=1)
    ph = st.slider('Enter ph : ',min_value=0,max_value=14,step=1)
    rain = st.slider('Enter rainfall : ',min_value=0 ,max_value=300,step=15)

    if st.button('Predict Result'):
        inp = np.array([n, p, k, temp, humidity, ph, rain]).reshape(-1, 7)
        output = model.predict(inp)[0]

        st.success('Suggested Crop : {}'. format(output))
        st.success('Yield : {}'. format(crops[output][0]))
        st.success('Required Watering :  {}'. format(crops[output][1]))
        st.success('Suggested Crop Pesticides : {}'. format(crops[output][2]))

if __name__ == '__main__':
    main()
            
