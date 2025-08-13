# Diabetes Classification Using RandomForest and Gradio

This project demonstrates a **machine learning pipeline** that classifies whether a person has diabetes based.
The classification is performed using a **RandomForestClassifier**, and the results are shared interactively using **Gradio**.

---

## 📝 Project Overview

- **Dataset:** `diabetes.csv`
- **Algorithm:** RandomForestClassifier (Scikit-learn)
- **Interface:** Gradio web interface
- **Objective:** Predict the likelihood of diabetes based on patient health measurements.

---

## 🔧 Features

- Load and preprocess the `diabetes.csv` dataset.
- Split data into training and testing sets.
- Train a **RandomForest classifier**.
- Evaluate the model with accuracy score.
- Deploy an interactive **Gradio interface** for users to input features and get predictions in real-time.

---

## 💻 Installation

Make sure you have Python 3.x installed. Then, install the required libraries:

```bash
pip install pandas scikit-learn gradio
🚀 Usage

python app.py
Open the Gradio interface in your browser (usually http://127.0.0.1:7860/) and enter the patient data to get predictions.

📊 Example
The Gradio interface allows you to input:
Pregnancies
Plasma Glucose
Diastolic Blood Pressure
Triceps Thickness
Serum Insulin
BMI
DiabetesPedigreeFunction
Age

…and outputs whether the patient is likely Diabetic or Non-Diabetic.

📈 Model Performance
Model: RandomForestClassifier

Accuracy: ~[92%]

⚙️ Technologies Used
Python – Programming language

Pandas – Data handling

Scikit-learn – Machine learning

Gradio – Web interface for ML models

📁 Dataset
The diabetes.csv file should be placed in the project folder.
📜 License
This project is open-source and available for anyone to use.

⭐ Contributing
Contributions are welcome! Feel free to submit pull requests or open issues.

📞 Contact
Author: Mohamed Tarek
GitHub: tarektech1st
@: mohamed.tarek1st@outlook.com
https://www.linkedin.com/in/mohamed-tarek-5a3599a0/
https://www.youtube.com/@TarekTech-u7l5d
https://t.me/+RxrtyTjEjdQzMjI0
