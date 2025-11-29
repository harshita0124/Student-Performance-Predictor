import streamlit as st
import numpy as np
import joblib # for loading model

st.set_page_config(
    page_title = "Performance Predictor",
    page_icon = "🎓"
)

st.title("🎓 Students Performance Prediction")
st.write("It is a simple student performance prediction ML Model, which is trained on Linear Regression Algorithm")
st.markdown("---")

# Sidebar

st.sidebar.title("Student Performance Tracker")
st.sidebar.image("Image_ml.webp")
st.sidebar.write("It is a simple Machine Learning based app, which is used for tracking and predicting the students performance using parameters such as:")
st.sidebar.markdown("- Gender")
st.sidebar.markdown("- Hours Studied")
st.sidebar.markdown("- Attendance Percentage")
st.sidebar.markdown("- Assignment Completed")

st.sidebar.markdown("---")
st.sidebar.markdown("`Made for you !!😀😀`")

# Load Model

model = joblib.load("model.pkl")

input_names = ["gender", "hours_studied", "attendance_percent", "assignments_completed"]
inputs = []

gender_value = st.selectbox("Gender", [0, 1])
hours_studied_values = st.number_input("Hours Studied", min_value = 5, max_value = 17)
attendance_percent_values = st.number_input("Attendance Percentage", min_value = 60, max_value = 99)
assignments_completed_values = st.number_input("Assignment Completed", min_value = 5, max_value = 14)

inputs = [gender_value, hours_studied_values, attendance_percent_values, assignments_completed_values ]

inputs = np.array(inputs).reshape(1, -1)

if st.button("predict"):
    output = model.predict(inputs)
    st.success(f"test_score {output}")

    if output < 33:
        st.warning("Student Failed ")
    elif output > 33 and output < 50:
        st.warning("You were on the brink of fail")
    else :
        st.success("You passed with good marks!!🎉🎉")


