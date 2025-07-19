import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and metadata
@st.cache_resource
def load_models():
    try:
        # Load classification model (predicts income category)
        clf_model = joblib.load("salary_classification_model.pkl")
        # Load regression model (estimates salary value)  
        reg_model = joblib.load("salary_regression_model.pkl")
        # Load label encoder
        label_encoder = joblib.load("label_encoder.pkl")
        # Load column info
        column_info = joblib.load("column_info.pkl")
        return clf_model, reg_model, label_encoder, column_info
    except FileNotFoundError:
        st.error("Model files not found! Please run 'python train_model.py' first.")
        st.stop()

# Load models
clf_model, reg_model, label_encoder, column_info = load_models()

# App title and description
st.title("🏢 Employee Salary Prediction System")
st.markdown("### Predict employee income category and estimated salary based on demographic data")

st.sidebar.markdown("## How to use:")
st.sidebar.markdown("1. Fill in all the employee details")
st.sidebar.markdown("2. Click 'Predict Income' to see results")
st.sidebar.markdown("3. Get both category prediction and salary estimate")

# Create input form
st.subheader("Employee Information")

# Create columns for better layout
col1, col2 = st.columns(2)

# Dictionary to store user inputs
user_input = {}

with col1:
    st.markdown("#### Personal Information")
    
    # Age
    user_input['age'] = st.number_input(
        "Age", 
        min_value=16, max_value=90, value=30,
        help="Employee's age in years"
    )
    
    # Gender
    if 'gender' in column_info['all_columns']:
        user_input['gender'] = st.selectbox(
            "Gender", 
            ["Male", "Female"]
        )
    elif 'sex' in column_info['all_columns']:
        user_input['sex'] = st.selectbox(
            "Gender", 
            ["Male", "Female"]
        )
    
    # Race
    if 'race' in column_info['all_columns']:
        user_input['race'] = st.selectbox(
            "Race", 
            ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
        )
    
    # Native Country
    if 'native-country' in column_info['all_columns']:
        user_input['native-country'] = st.selectbox(
            "Native Country",
            ["United-States", "Mexico", "Philippines", "Germany", "Puerto-Rico", 
             "Canada", "El-Salvador", "India", "Cuba", "England", "China", "Other"]
        )

with col2:
    st.markdown("#### Work & Education")
    
    # Work Class
    if 'workclass' in column_info['all_columns']:
        user_input['workclass'] = st.selectbox(
            "Work Class",
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
             "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        )
    
    # Education
    if 'education' in column_info['all_columns']:
        user_input['education'] = st.selectbox(
            "Education Level",
            ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
             "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
             "5th-6th", "10th", "1st-4th", "Preschool", "12th"]
        )
    
    # Education Number
    if 'educational-num' in column_info['all_columns']:
        user_input['educational-num'] = st.slider(
            "Education Years", 
            min_value=1, max_value=16, value=10,
            help="Number of years of education"
        )
    elif 'education-num' in column_info['all_columns']:
        user_input['education-num'] = st.slider(
            "Education Years", 
            min_value=1, max_value=16, value=10,
            help="Number of years of education"
        )

# Second row of inputs
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Employment Details")
    
    # Occupation
    if 'occupation' in column_info['all_columns']:
        user_input['occupation'] = st.selectbox(
            "Occupation",
            ["Tech-support", "Craft-repair", "Other-service", "Sales", 
             "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
             "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
             "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        )
    
    # Hours per week
    if 'hours-per-week' in column_info['all_columns']:
        user_input['hours-per-week'] = st.slider(
            "Hours per Week", 
            min_value=1, max_value=100, value=40,
            help="Average hours worked per week"
        )

with col4:
    st.markdown("#### Personal Status")
    
    # Marital Status
    if 'marital-status' in column_info['all_columns']:
        user_input['marital-status'] = st.selectbox(
            "Marital Status",
            ["Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent",
             "Separated", "Married-AF-spouse", "Widowed"]
        )
    
    # Relationship
    if 'relationship' in column_info['all_columns']:
        user_input['relationship'] = st.selectbox(
            "Relationship",
            ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"]
        )

# Financial information
st.markdown("#### Financial Information")
col5, col6 = st.columns(2)

with col5:
    if 'capital-gain' in column_info['all_columns']:
        user_input['capital-gain'] = st.number_input(
            "Capital Gain", 
            min_value=0, max_value=100000, value=0,
            help="Capital gains from investments"
        )

with col6:
    if 'capital-loss' in column_info['all_columns']:
        user_input['capital-loss'] = st.number_input(
            "Capital Loss", 
            min_value=0, max_value=10000, value=0,
            help="Capital losses from investments"
        )

# Prediction section
st.markdown("---")
if st.button("🔮 Predict Income", type="primary", use_container_width=True):
    
    # Create input dataframe
    input_df = pd.DataFrame([user_input])
    
    # Ensure all expected columns are present
    for col in column_info['all_columns']:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value for missing columns
    
    # Reorder columns to match training data
    input_df = input_df[column_info['all_columns']]
    
    try:
        # Make classification prediction
        class_prediction = clf_model.predict(input_df)[0]
        class_prob = clf_model.predict_proba(input_df)[0]
        predicted_category = label_encoder.inverse_transform([class_prediction])[0]
        
        # Make regression prediction (salary estimate)
        salary_prediction = reg_model.predict(input_df)[0]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Create result columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("### 📊 Income Category")
            if "<=50K" in predicted_category:
                st.markdown(f"<h2 style='color: orange'>🔸 {predicted_category}</h2>", unsafe_allow_html=True)
                confidence = max(class_prob) * 100
                st.markdown(f"**Confidence:** {confidence:.1f}%")
            else:
                st.markdown(f"<h2 style='color: green'>🔹 {predicted_category}</h2>", unsafe_allow_html=True)
                confidence = max(class_prob) * 100
                st.markdown(f"**Confidence:** {confidence:.1f}%")
        
        with result_col2:
            st.markdown("### 💰 Estimated Salary")
            st.markdown(f"<h2 style='color: blue'>${salary_prediction:,.0f}</h2>", unsafe_allow_html=True)
            st.markdown("*Annual estimated salary*")
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 📈 Prediction Details")
        
        # Show probability breakdown
        prob_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability': class_prob
        }).sort_values('Probability', ascending=False)
        
        st.markdown("**Category Probabilities:**")
        for _, row in prob_df.iterrows():
            st.write(f"• {row['Category']}: {row['Probability']:.1%}")
            
        # Salary range information
        st.markdown("**Salary Estimate Explanation:**")
        if "<=50K" in predicted_category:
            st.info("💡 Based on the prediction model, this profile typically falls in the lower income category. The estimated salary represents an average for this category.")
        else:
            st.info("💡 Based on the prediction model, this profile typically falls in the higher income category. The estimated salary represents an average for this category.")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Please check that all required fields are filled correctly.")

# Footer
