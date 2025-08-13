#pip install pandas read files
#pip install gradio view interface
#pip install scikit-learn Algo, traing - test-spliting-evaluation metrics (ML wordflow )

import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load and prepare the dataset
# -------------------------------

# Load dataset from CSV file
df = pd.read_csv("diabetes.csv")

# Remove 'PatientID' column as it's not a useful feature for prediction
df = df.drop(columns=["PatientID"])

# Separate features (X) and target variable (y)
X = df.drop(columns=["Diabetic"])   # Input variables
y = df["Diabetic"]                  # Output/target variable


# -------------------------------
# 2. Split data into training and testing sets
# -------------------------------

# Split data into 80% training and 20% testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)


# -------------------------------
# 3. Train the machine learning model
# -------------------------------

# Create a Random Forest Classifier (ensemble decision tree model)
model = RandomForestClassifier(random_state=42)

# Train the model on training data
model.fit(X_train, y_train)


# -------------------------------
# 4. Evaluate model accuracy
# -------------------------------

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy score
model_accuracy = accuracy_score(y_test, y_pred)


# -------------------------------
# 5. Define prediction function
# -------------------------------

def predict(Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
            SerumInsulin, BMI, DiabetesPedigree, Age):
    """
    Predict if a person is diabetic based on their health metrics.
    Returns prediction (diabetic / not diabetic) and model accuracy.
    """
    try:
        # Prepare the input data in the correct shape for the model
        input_data = [[
            Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
            SerumInsulin, BMI, DiabetesPedigree, Age
        ]]

        # Make prediction using trained model
        pred = model.predict(input_data)[0]

        # Convert numeric prediction to readable label
        result = "🟢 Diabetic" if pred == 1 else "🔵 Not Diabetic"

        # Return prediction and model accuracy
        return f"{result}\n\n📊 Model Accuracy: {model_accuracy * 100:.2f}%"

    except Exception as e:
        # Handle errors (e.g., wrong input type)
        return f"❌ Error: {e}"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------------
# 6. Create Gradio interface
# -------------------------------

# Build interactive UI for entering patient data and getting predictions
iface = gr.Interface(
    fn=predict,  # Function to call when user submits input
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Plasma Glucose"),
        gr.Number(label="Diastolic Blood Pressure"),
        gr.Number(label="Triceps Thickness"),
        gr.Number(label="Serum Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Text(label="Prediction + Accuracy"),
    title="Diabetes Classifier",
    description="Enter health metrics to predict if a person is diabetic. Includes model accuracy.",
)

# Launch the app in a local web browser
iface.launch()
