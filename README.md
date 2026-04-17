# Groundwater & Soil Moisture Analytics Dashboard

This project elevates exploratory groundwater data into a professional **Business Intelligence (BI) and Predictive Dashboard**. It focuses on domain-specific feature engineering and interactive data visualization.

## Professional Features
- **Interactive Dashboard:** Built with **Streamlit** and **Plotly** for high-quality, interactive visualizations.
- **Feature Engineering:** Implemented domain-specific metrics like *Usage Intensity* and *Recharge Efficiency* to improve model robustness.
- **Scenario Analysis:** Features a "What-If" prediction tool that allows users to simulate different environmental scenarios and see real-time impact on water stress levels.
- **Model Explainability:** Includes a "Model Insights" section that visualizes feature importance, explaining the "Why" behind predictions.

## Project Structure
```text
Project Soil moisture/
├── app.py                  # Streamlit Dashboard Script
├── train.py                # Model Training with Feature Engineering
├── requirements.txt        # Dashboard Dependencies
├── Groundwater.csv         # Raw Dataset
├── groundwater_model.pkl   # Trained RF Model (generated)
├── groundwater_scaler.pkl  # Fitted Scaler (generated)
└── feature_names.pkl       # Serialized feature list (generated)
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Predictive Model
This will perform the feature engineering and save the model artifacts.
```bash
python train.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

## Technical Highlights
- **Model:** Random Forest Classifier.
- **Engineering:** Calculated ratios of usage-to-availability to better capture "stress" than raw volume alone.
- **Visualization:** Used Plotly for dynamic charts that allow for zooming and filtering by state/situation.
