import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load datasets
drug_df = pd.read_csv("Drug.csv")
training_df = pd.read_csv("Training.csv")

# Load label encoder for disease names
label_encoder = LabelEncoder()
label_encoder.fit(training_df['prognosis'])

# Load model (assuming it is trained and saved)
model = RandomForestClassifier()  # Replace this with the actual loading of your model
model.fit(training_df.drop(columns=['prognosis']), label_encoder.transform(training_df['prognosis']))

# Title of the application
st.title("Disease Detection and Drug Recommendation System")

# Option to choose action
action = st.selectbox("Choose an action", ["Disease Prediction", "Medication Recommendation", "View & Filter Datasets"])

if action == "Disease Prediction":
    st.subheader("Disease Prediction based on Symptoms")
    symptoms_input = st.multiselect("Select Symptoms", options=training_df.columns[:-1])  # Exclude target column
    input_data = [1 if symptom in symptoms_input else 0 for symptom in training_df.columns[:-1]]

    # Button for disease prediction
    if st.button("Predict Disease"):
        input_df = pd.DataFrame([input_data], columns=training_df.columns[:-1])
        disease_prediction_probs = model.predict_proba(input_df)
        disease_predictions = model.predict(input_df)

        # Create a DataFrame for probabilities
        probabilities_df = pd.DataFrame(disease_prediction_probs, columns=label_encoder.classes_)

        # Combine predictions with probabilities and percentage
        results_df = pd.DataFrame({
            'Disease': label_encoder.inverse_transform(range(len(probabilities_df.columns))),
            'Probability': probabilities_df.values.flatten(),
            'Percentage': probabilities_df.values.flatten() * 100
        })

        # Sort by probability and select top 10 diseases
        ranked_results = results_df.sort_values(by='Probability', ascending=False).head(10)

        # Display the ranked diseases with their probabilities and percentages
        st.subheader("Top 10 Probable Diseases:")
        st.table(ranked_results[['Disease', 'Probability', 'Percentage']])

elif action == "Medication Recommendation":
    # User input for Gender, Age, and Disease
    gender = st.selectbox("Select Gender", options=['Male', 'Female'])
    age = st.number_input("Enter Age", min_value=1, max_value=120, step=1)

    disease_input = st.selectbox("Select Disease", options=drug_df['Disease'].unique())

    # Button for medication recommendation
    if st.button("Recommend Medication"):
        # Convert Gender input to numeric
        gender_encoded = 0 if gender == 'Male' else 1

        # Check for exact matches first
        exact_match = drug_df[
            (drug_df['Disease'] == disease_input) &
            (drug_df['Gender'] == gender) &
            (drug_df['Age'] == age)
        ]

        if not exact_match.empty:
            st.subheader("Recommended Medications (Exact Match):")
            st.table(exact_match[['Drug', 'Disease', 'Gender', 'Age']])
        else:
            # If no exact match is found, look for matches based on gender and disease
            fallback_match = drug_df[
                (drug_df['Disease'] == disease_input) &
                (drug_df['Gender'] == gender)
            ]

            if not fallback_match.empty:
                st.subheader("Recommended Medications (Fallback Match):")
                st.table(fallback_match[['Drug', 'Disease', 'Gender', 'Age']])
            else:
                # If no matches for disease and gender, find the closest age match
                closest_age_matches = drug_df[drug_df['Disease'] == disease_input]

                if not closest_age_matches.empty:
                    # Get the unique ages available for the specific disease
                    available_ages = closest_age_matches['Age'].unique()

                    # Find the closest age in the available ages
                    closest_age = min(available_ages, key=lambda x: abs(x - age))

                    # Get drugs associated with the closest available age
                    closest_age_drugs = closest_age_matches[closest_age_matches['Age'] == closest_age]

                    st.subheader("Recommended Medications (Closest Age Match):")
                    st.table(closest_age_drugs[['Drug', 'Disease', 'Gender', 'Age']])
                else:
                    # If no matches found, return all drugs related to the disease regardless of age
                    all_drugs_for_disease = drug_df[drug_df['Disease'] == disease_input]

                    st.subheader("Recommended Medications (All Available for Disease):")
                    st.table(all_drugs_for_disease[['Drug', 'Disease', 'Gender', 'Age']])

elif action == "View & Filter Datasets":
    dataset_option = st.selectbox("Choose Dataset to View and Filter", ["Drug Dataset", "Training Dataset"])

    if dataset_option == "Drug Dataset":
        st.subheader("Drug Dataset with Filters")
        disease_filter = st.selectbox("Filter by Disease", options=drug_df['Disease'].unique())
        gender_filter = st.selectbox("Filter by Gender", options=drug_df['Gender'].unique())
        
        # Replaced slider with text inputs for min and max age
        min_age = st.text_input("Enter Min Age", value=str(int(drug_df['Age'].min())))
        max_age = st.text_input("Enter Max Age", value=str(int(drug_df['Age'].max())))

        # Convert text inputs to integers
        try:
            min_age = int(min_age)
            max_age = int(max_age)
        except ValueError:
            st.error("Please enter valid numbers for age")

        filtered_drug_df = drug_df[
            (drug_df['Disease'] == disease_filter) &
            (drug_df['Gender'] == gender_filter) &
            (drug_df['Age'] >= min_age) &
            (drug_df['Age'] <= max_age)
        ]

        st.dataframe(filtered_drug_df)

    elif dataset_option == "Training Dataset":
        st.subheader("Training Dataset with Filters")
        prognosis_filter = st.selectbox("Filter by Prognosis", options=training_df['prognosis'].unique())

        # Apply filter to show only rows with the selected prognosis
        filtered_training_df = training_df[training_df['prognosis'] == prognosis_filter]

        # Reorder columns to bring 'prognosis' to the start
        columns = ['prognosis'] + [col for col in filtered_training_df.columns if col != 'prognosis']
        filtered_training_df = filtered_training_df[columns]

        # For each row, only show columns with value 1
        filtered_training_df_only_ones = filtered_training_df.apply(lambda row: row[row == 1], axis=1).dropna(how='all', axis=1)

        # Ensure the 'prognosis' column is included at the start
        filtered_training_df_only_ones.insert(0, 'prognosis', filtered_training_df['prognosis'])

        # Display the filtered dataset with only the columns having 1
        st.dataframe(filtered_training_df_only_ones)

# Button to show the full datasets without filters
if st.button("Show Full Datasets"):
    st.subheader("Full Drug Dataset")
    st.dataframe(drug_df)

    st.subheader("Full Training Dataset")
    st.dataframe(training_df)
