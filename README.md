# Disease Prediction and Drug Recommendation System

## Overview

The **Disease Prediction and Drug Recommendation System** is a Streamlit-based web application that predicts diseases based on user-provided symptoms and recommends appropriate medications. The system utilizes machine learning techniques, specifically a **Random Forest Classifier**, trained on a dataset of symptoms and diseases. Additionally, it provides functionalities to view and filter datasets, check medication timing, and update drug information.

## Features

- **Disease Prediction**: Users can select symptoms, and the system predicts probable diseases along with their probabilities.
- **Medication Recommendation**: Based on disease, age, and gender, the system suggests relevant medications.
- **Medication Timing**: Users can check the prescription timing for a selected drug.
- **Dataset Viewing & Filtering**: Users can explore and filter the drug and training datasets.
- **Dataset Modification**: Ability to update or add new drug entries to the database.

## Technologies Used

- **Python**
- **Streamlit** (for building the web application)
- **Scikit-learn** (for machine learning model)
- **Pandas** (for data handling and preprocessing)
- **NumPy** (for numerical operations)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/adityakarri12/Disease-Prediction-and-Drug-recommendations-System.git
   cd Disease-Prediction-and-Drug-recommendations-System
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**

   ```bash
   streamlit run disease_app.py
   ```

## Dataset Information

### **Training Dataset (********`Training.csv`********\*\*\*\*\*\*\*\*)**

This dataset contains symptoms as binary features (`1` for presence, `0` for absence) and a `prognosis` column indicating the diagnosed disease.

### **Drug Dataset (********`Drug.csv`********\*\*\*\*\*\*\*\*)**

This dataset consists of:

- **Drug Name**
- **Disease it treats**
- **Gender suitability**
- **Age range**
- **Prescription timing**

## How It Works

### **Disease Prediction**

1. Select symptoms from a dropdown list.
2. The model predicts possible diseases based on the symptoms.
3. The top 10 probable diseases are displayed with confidence percentages.

### **Medication Recommendation**

1. Choose gender and enter age.
2. Select the diagnosed disease.
3. The system suggests relevant medications based on the dataset.

### **Viewing and Filtering Datasets**

- Users can filter the **Drug dataset** based on disease, gender, and age range.
- Users can filter the **Training dataset** based on prognosis.

### **Updating Datasets**

- Modify existing drug records by selecting a row index and updating values.
- Add new drug records with disease, gender, age, and drug name details.

## File Structure

```
📁 disease-prediction
│── app.py                 # Main Streamlit application file
│── requirements.txt        # Required Python packages
│── Training.csv            # Training dataset for disease prediction
│── Drug.csv                # Medication dataset
│── README.md               # Project documentation
```

## Requirements

The `requirements.txt` file includes:

```
streamlit
pandas
scikit-learn
numpy>=1.21.0,<2.0.0
```

## Future Improvements

- Enhance model accuracy with a larger dataset.
- Implement an advanced **NLP-based symptom search**.
- Include **user authentication** for personalized recommendations.
- Add a **chatbot assistant** for interactive healthcare guidance.

## Contributing

Feel free to fork this repository and submit pull requests with improvements.

## License

This project is licensed under the **MIT License**.

---

**Author:** Karri Aditya

**inkedIn:** [linkedin.com/in/aditya-karri-7128a61b1](https://linkedin.com/in/aditya-karri-7128a61b1)\
**LeetCode:** [leetcode.com/u/karri\_aditya/](https://leetcode.com/u/karri_aditya/)

