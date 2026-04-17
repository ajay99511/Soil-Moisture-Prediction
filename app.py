import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Groundwater Analytics Dashboard", layout="wide")

# Load data and model artifacts
@st.cache_data
def load_data():
    return pd.read_csv('Groundwater.csv')

def load_artifacts():
    model = joblib.load('groundwater_model.pkl')
    scaler = joblib.load('groundwater_scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

st.title("💧 Groundwater & Soil Moisture Analytics")
st.markdown("""
    This dashboard provides a professional-grade analysis of groundwater availability across Indian states. 
    It uses a **Random Forest Classifier** with engineered features to predict water stress levels.
""")

df = load_data()

# Sidebar for Navigation
menu = st.sidebar.selectbox("Menu", ["Data Explorer", "Stress Prediction", "Model Insights"])

if menu == "Data Explorer":
    st.header("🔍 Dataset Exploration")
    st.dataframe(df.style.background_gradient(cmap='Blues'))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Situations")
        fig = px.pie(df, names='Situation', hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Rainfall vs Usage")
        fig = px.scatter(df, x='Total_Rainfall', y='Total_Usage', color='Situation', size='Net annual groundwater availability', hover_name='States')
        st.plotly_chart(fig)

elif menu == "Stress Prediction":
    st.header("🔮 Scenario Prediction (What-If Analysis)")
    
    if not os.path.exists('groundwater_model.pkl'):
        st.error("Please run 'train.py' first to generate the model!")
    else:
        model, scaler, features = load_artifacts()
        
        st.write("Adjust the parameters below to see how the Groundwater Situation changes.")
        
        col1, col2, col3 = st.columns(3)
        
        # User inputs based on core features
        with col1:
            rain_m = st.slider("Monsoon Rainfall Recharge", 0.0, 50.0, 15.0)
            rain_nm = st.slider("Non-Monsoon Rainfall Recharge", 0.0, 20.0, 5.0)
            total_rain = st.slider("Total Annual Rainfall", 0.0, 100.0, 30.0)
            
        with col2:
            usage_irr = st.slider("Irrigation Usage", 0.0, 50.0, 10.0)
            usage_dom = st.slider("Domestic/Industrial Usage", 0.0, 10.0, 2.0)
            discharge = st.slider("Natural Discharge", 0.0, 10.0, 2.0)

        with col3:
            avail = st.slider("Annual Availability", 1.0, 100.0, 25.0)
            other_recharge = 5.0 # Constant for simplicity
            proj_demand = 3.0
            future_irr = 15.0

        # Create input array matching the EXACT order of training features
        # X order: [Recharge from rainfallMonsoon season, Recharge from other sources, Recharge from rainfallNon-monsoon season, 
        #           Recharge from other sources.1, Total_Rainfall, Natural discharge during non-monsoon season, 
        #           Net annual groundwater availability, Irrigation, Domestic and industrial uses, Total_Usage, 
        #           Projected demand for domestic and industrial uses upto 2025, Groundwater availability for future irrigation use,
        #           Usage_Intensity, Recharge_Efficiency]
        
        total_usage = usage_irr + usage_dom
        usage_intensity = total_usage / (avail + 1e-5)
        recharge_eff = (rain_m + rain_nm) / (total_rain + 1e-5)

        input_dict = {
            'Recharge from rainfallMonsoon season': rain_m,
            'Recharge from other sources': other_recharge,
            'Recharge from rainfallNon-monsoon season': rain_nm,
            'Recharge from other sources.1': other_recharge,
            'Total_Rainfall': total_rain,
            'Natural discharge during non-monsoon season': discharge,
            'Net annual groundwater availability': avail,
            'Irrigation': usage_irr,
            'Domestic and industrial uses': usage_dom,
            'Total_Usage': total_usage,
            'Projected demand for domestic and industrial uses upto 2025': proj_demand,
            'Groundwater availability for future irrigation use': future_irr,
            'Usage_Intensity': usage_intensity,
            'Recharge_Efficiency': recharge_eff
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Scale and Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Predicted Situation", prediction)
            if prediction == "EXCESS":
                st.success("The state has sustainable groundwater levels.")
            elif prediction == "MODERATED":
                st.warning("Groundwater usage is approaching critical levels.")
            else:
                st.error("WARNING: Semicritical or Critical groundwater levels detected.")
                
        with res_col2:
            prob_df = pd.DataFrame({'Category': model.classes_, 'Probability': probs})
            fig = px.bar(prob_df, x='Category', y='Probability', title="Model Confidence")
            st.plotly_chart(fig, use_container_width=True)

elif menu == "Model Insights":
    st.header("🧠 Technical Explanation")
    if os.path.exists('groundwater_model.pkl'):
        model, _, features = load_artifacts()
        
        st.subheader("Feature Importance")
        st.write("Which factors contribute most to the groundwater situation?")
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig = px.bar(importances, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig)
        
        st.info("""
            **Technical Note:** We implemented **Feature Engineering** by calculating 'Usage Intensity' and 'Recharge Efficiency'. 
            Notice how these engineered features often rank higher in importance than raw data points!
        """)
