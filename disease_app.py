import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load datasets
drug_df = pd.read_csv("Drug.csv")
training_df = pd.read_csv("Training.csv")

# Encode disease names
label_encoder = LabelEncoder()
label_encoder.fit(training_df['prognosis'])

# Load and train model
model = RandomForestClassifier()
model.fit(training_df.drop(columns=['prognosis']), label_encoder.transform(training_df['prognosis']))

# App title
st.title("Disease Prediction and Drug Recommendation System")

# Sidebar for navigation
st.sidebar.title("Navigation")
action = st.sidebar.radio(
    "Choose an action",
    ["Disease Prediction", "Medication Recommendation", "View Medication Timing", "View & Filter Datasets"]
)

# Disease Prediction Section
if action == "Disease Prediction":
    st.header("Disease Prediction based on Symptoms")
    symptoms_input = st.multiselect("Select Symptoms", options=training_df.columns[:-1])
    input_data = [1 if symptom in symptoms_input else 0 for symptom in training_df.columns[:-1]]

    if st.button("Predict Disease"):
        input_df = pd.DataFrame([input_data], columns=training_df.columns[:-1])
        disease_prediction_probs = model.predict_proba(input_df)
        disease_predictions = model.predict(input_df)
        
        # Create DataFrame for probabilities
        probabilities_df = pd.DataFrame(disease_prediction_probs, columns=label_encoder.classes_)
        results_df = pd.DataFrame({
            'Disease': label_encoder.inverse_transform(range(len(probabilities_df.columns))),
            'Probability': probabilities_df.values.flatten(),
            'Percentage': probabilities_df.values.flatten() * 100
        }).sort_values(by='Probability', ascending=False).head(10)
        
        st.subheader("Top 10 Probable Diseases:")
        st.table(results_df[['Disease', 'Probability', 'Percentage']])

# Medication Recommendation Section
elif action == "Medication Recommendation":
    st.header("Medication Recommendation")
    gender = st.selectbox("Select Gender", options=['Male', 'Female'])
    age = st.number_input("Enter Age", min_value=1, max_value=120, step=1)
    disease_input = st.selectbox("Select Disease", options=drug_df['Disease'].unique())

    if st.button("Recommend Medication"):
        gender_encoded = 0 if gender == 'Male' else 1
        exact_match = drug_df[(drug_df['Disease'] == disease_input) & (drug_df['Gender'] == gender) & (drug_df['Age'] == age)]

        if not exact_match.empty:
            st.subheader("Recommended Medications (Exact Match):")
            st.table(exact_match[['Drug', 'Disease', 'Gender', 'Age']])
        else:
            fallback_match = drug_df[(drug_df['Disease'] == disease_input) & (drug_df['Gender'] == gender)]
            if not fallback_match.empty:
                st.subheader("Recommended Medications (Fallback Match):")
                st.table(fallback_match[['Drug', 'Disease', 'Gender', 'Age']])
            else:
                closest_age_matches = drug_df[drug_df['Disease'] == disease_input]
                if not closest_age_matches.empty:
                    available_ages = closest_age_matches['Age'].unique()
                    closest_age = min(available_ages, key=lambda x: abs(x - age))
                    closest_age_drugs = closest_age_matches[closest_age_matches['Age'] == closest_age]
                    st.subheader("Recommended Medications (Closest Age Match):")
                    st.table(closest_age_drugs[['Drug', 'Disease', 'Gender', 'Age']])
                else:
                    all_drugs_for_disease = drug_df[drug_df['Disease'] == disease_input]
                    st.subheader("Recommended Medications (All Available for Disease):")
                    st.table(all_drugs_for_disease[['Drug', 'Disease', 'Gender', 'Age']])

# View Medication Timing Section
elif action == "View Medication Timing":
    st.header("View Timing for Medications")
    selected_drug = st.selectbox("Select a Medicine", options=drug_df['Drug'].unique())
    
    # Retrieve the prescription timing for the selected drug
    timing_info = drug_df[drug_df['Drug'] == selected_drug]['Prescription'].iloc[0]
    st.subheader(f"Timing to take {selected_drug}:")
    st.write(f"The medication should be taken **{timing_info}**.")

# View & Filter Datasets Section
elif action == "View & Filter Datasets":
    dataset_option = st.selectbox("Choose Dataset to View and Filter", ["Drug Dataset", "Training Dataset"])

    if dataset_option == "Drug Dataset":
        st.header("Drug Dataset with Filters")
        disease_filter = st.selectbox("Filter by Disease", options=drug_df['Disease'].unique())
        gender_filter = st.selectbox("Filter by Gender", options=drug_df['Gender'].unique())
        min_age = st.text_input("Enter Min Age", value=str(int(drug_df['Age'].min())))
        max_age = st.text_input("Enter Max Age", value=str(int(drug_df['Age'].max())))

        try:
            min_age = int(min_age)
            max_age = int(max_age)
        except ValueError:
            st.error("Please enter valid numbers for age")

        filtered_drug_df = drug_df[(drug_df['Disease'] == disease_filter) & 
                                   (drug_df['Gender'] == gender_filter) & 
                                   (drug_df['Age'] >= min_age) & 
                                   (drug_df['Age'] <= max_age)]
        st.dataframe(filtered_drug_df)

        # Add option to update the dataset with new rows
        st.subheader("Add or Update Rows in Drug Dataset")
        
        # Update row inputs
        st.markdown("### Update Existing Row")
        row_to_update = st.number_input("Row Index to Update", min_value=0, max_value=len(drug_df)-1, step=1)
        updated_disease = st.text_input("Updated Disease", value=drug_df.loc[row_to_update, 'Disease'])
        updated_gender = st.selectbox("Updated Gender", ['Male', 'Female'], index=['Male', 'Female'].index(drug_df.loc[row_to_update, 'Gender']))
        updated_age = st.number_input("Updated Age", min_value=15, max_value=50, value=int(drug_df.loc[row_to_update, 'Age']))
        updated_drug = st.text_input("Updated Drug", value=drug_df.loc[row_to_update, 'Drug'])
        
        if st.button("Update Row"):
            drug_df.at[row_to_update, 'Disease'] = updated_disease
            drug_df.at[row_to_update, 'Gender'] = updated_gender
            drug_df.at[row_to_update, 'Age'] = updated_age
            drug_df.at[row_to_update, 'Drug'] = updated_drug
            drug_df.to_csv("Drug.csv", index=False)  # Save changes
            st.success("Row updated successfully!")
            st.dataframe(drug_df)
        
        # Add new row inputs
        st.markdown("### Add New Row")
        new_disease = st.text_input("Disease")
        new_gender = st.selectbox("Gender", ['Male', 'Female'])
        new_age = st.number_input("Age", min_value=15, max_value=50)
        new_drug = st.text_input("Drug")
        
        if st.button("Add Row"):
            new_row = pd.DataFrame({'Disease': [new_disease], 'Gender': [new_gender], 'Age': [new_age], 'Drug': [new_drug]})
            drug_df = pd.concat([drug_df, new_row], ignore_index=True).sort_values(by='Disease').reset_index(drop=True)
            drug_df.to_csv("Drug.csv", index=False)  # Save changes
            st.success("Row added successfully!")
            st.dataframe(drug_df)

    elif dataset_option == "Training Dataset":
        st.header("Training Dataset with Filters")
        prognosis_filter = st.selectbox("Filter by Prognosis", options=training_df['prognosis'].unique())
        filtered_training_df = training_df[training_df['prognosis'] == prognosis_filter]
        columns = ['prognosis'] + [col for col in filtered_training_df.columns if col != 'prognosis']
        filtered_training_df = filtered_training_df[columns]
        filtered_training_df_only_ones = filtered_training_df.apply(lambda row: row[row == 1], axis=1).dropna(how='all', axis=1)
        filtered_training_df_only_ones.insert(0, 'prognosis', filtered_training_df['prognosis'])
        st.dataframe(filtered_training_df_only_ones)

# Button to show the full datasets without filters
if st.button("Show Full Datasets"):
    st.header("Full Drug Dataset")
    st.dataframe(drug_df)
    st.header("Full Training Dataset")
    st.dataframe(training_df)
