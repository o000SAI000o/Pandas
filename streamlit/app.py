import streamlit as st

#2.1: First App
st.title("My First Streamlit App")
st.write("Hello from Streamlit!")

#2.2 user inputs
name = st.text_input("Enter your name")
if st.button("greet"):
    st.write(f"Hello dear {name} !")
    
#2.3 charts and visuals
import pandas as pd
import matplotlib.pyplot as plt

st.line_chart(pd.Series([1,2,3,4,5]))

fig,ax = plt.subplots()
ax.plot([1,2,3,4,5],[4,1,2,3,5])
st.pyplot(fig)

#2.4 ML model integration
import joblib
model = joblib.load('titanic_pipeline.pkl')

#user input
age = st.number_input("Enter age -",min_value=0, max_value=100,value=29)  
fare = st.number_input("Enter fare -",min_value=0, max_value=30,value=29)  
gender = st.selectbox("Select gender -",['male','female'])  

# Create input DataFrame with the required column names
df = pd.DataFrame({
    'age':[age],
    'fare':[fare],
    'gender':[gender]
})

# Predict button
if st.button("predict ->"):
    prediction = model.predict(df)
    st.write("✅ prediction :","survived" if prediction[0]== 1 else "not survived")
    

