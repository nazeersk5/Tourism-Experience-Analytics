import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Tourism Analytics System", layout="wide")

# Upload Excel data instead of MySQL
@st.cache_data(show_spinner="Loading tourism data...")
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # Required columns
    required_columns = [
        'UserId', 'VisitYear', 'VisitMonth', 'AttractionId', 'Rating',
        'ContinentId', 'RegionId', 'CountryId', 'CityId', 'Country',
        'Region', 'Continent', 'AttractionTypeId', 'Attraction',
        'AttractionAddress', 'AttractionType', 'VisitModeId', 'VisitMode'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in Excel file: {missing_cols}")
        return None

    df = df.dropna(subset=['UserId', 'AttractionId', 'Rating'])

    # Add synthetic user profiles
    np.random.seed(42)
    unique_users = df['UserId'].unique()
    user_profiles = pd.DataFrame({
        'UserId': unique_users,
        'Age': np.random.randint(18, 70, len(unique_users)),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], len(unique_users)),
        'IncomeLevel': np.random.choice(['Low', 'Medium', 'High'], len(unique_users))
    })
    df = df.merge(user_profiles, on='UserId')
    return df

# Train visit mode prediction model
@st.cache_resource
def train_visit_mode_model(df):
    features = df[['Age', 'Gender', 'IncomeLevel', 'Country']]
    features = pd.get_dummies(features)
    le = LabelEncoder()
    target = le.fit_transform(df['VisitMode'])
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, le

# Recommendation system
def get_recommendations(df, user_id=None, country=None, visit_mode=None, num_recommendations=10):
    df = df.dropna(subset=['UserId', 'AttractionId', 'Rating'])
    rating_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)

    if user_id and user_id in rating_matrix.index:
        user_vec = rating_matrix.loc[user_id].values.reshape(1, -1)
        similarity = cosine_similarity(user_vec, rating_matrix.values)[0]
        similar_users = pd.Series(similarity, index=rating_matrix.index).sort_values(ascending=False)[1:11]
        sim_users_df = df[df['UserId'].isin(similar_users.index)]
        recommendations = sim_users_df.groupby('AttractionId').agg({
            'Rating': 'mean',
            'Attraction': 'first',
            'AttractionType': 'first',
            'AttractionAddress': 'first',
            'Country': 'first',
            'VisitMode': 'first'
        }).reset_index()
    else:
        recommendations = df.copy()

    if country:
        recommendations = recommendations[
            recommendations['Country'].fillna('').str.lower() == country.lower()
        ]
    if visit_mode:
        recommendations = recommendations[
            recommendations['VisitMode'].fillna('').str.lower() == visit_mode.lower()
        ]

    recommendations = (
        recommendations.groupby('AttractionId')
        .agg({
            'Attraction': 'first',
            'AttractionType': 'first',
            'AttractionAddress': 'first',
            'Rating': 'mean'
        })
        .sort_values(by='Rating', ascending=False)
        .head(num_recommendations)
    )

    return recommendations

def main():
    st.title("📊 Tourism Analytics & Recommendation System")

    uploaded_file = st.file_uploader("Upload Excel File with Tourism Data", type=["xlsx"])
    if uploaded_file is None:
        st.warning("Please upload an Excel file to continue.")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the section", ["Trend Analysis", "Visit Mode Prediction", "Personalized Recommendations"])

    if app_mode == "Trend Analysis":
        st.header("Tourism Trend Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("User Demographics")
            fig, ax = plt.subplots()
            sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue')
            ax.set_title("Age Distribution", fontsize=14)
            st.pyplot(fig)

            gender_counts = df['Gender'].value_counts()
            fig, ax = plt.subplots()
            gender_counts.plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
            ax.set_ylabel('')
            ax.set_title("Gender Distribution", fontsize=14)
            st.pyplot(fig)

        with col2:
            top_attractions = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x=top_attractions.values, y=top_attractions.index, ax=ax, palette='viridis')
            ax.set_xlabel("Average Rating")
            ax.set_ylabel("Attraction")
            ax.set_title("Top Rated Attractions")
            st.pyplot(fig)

            attraction_types = df['AttractionType'].value_counts().head(10)
            fig, ax = plt.subplots()
            attraction_types.plot.bar(ax=ax, color='coral')
            ax.set_ylabel("Count")
            ax.set_title("Popular Attraction Types")
            st.pyplot(fig)

        st.subheader("Visit Modes by Region")
        visit_region = pd.crosstab(df['VisitMode'], df['Region'], normalize='index')
        fig, ax = plt.subplots(figsize=(14, 6))
        visit_region.plot.bar(stacked=True, ax=ax, colormap='tab20')
        ax.set_ylabel("Proportion")
        ax.set_title("Visit Modes by Region")
        ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    elif app_mode == "Visit Mode Prediction":
        st.header("Visit Mode Prediction")
        model, le = train_visit_mode_model(df)

        with st.form("user_profile"):
            st.subheader("User Profile")
            age = st.slider("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            income = st.selectbox("Income Level", ['Low', 'Medium', 'High'])
            country = st.selectbox("Country", df['Country'].unique())

            if st.form_submit_button("Predict Visit Mode"):
                input_data = pd.DataFrame([[age, gender, income, country]], columns=['Age', 'Gender', 'IncomeLevel', 'Country'])
                input_data = pd.get_dummies(input_data)
                train_cols = model.feature_names_in_
                for col in train_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[train_cols]
                prediction = model.predict(input_data)
                predicted_mode = le.inverse_transform(prediction)[0]
                st.success(f"Predicted Visit Mode: **{predicted_mode}**")

                importances = pd.DataFrame({
                    'Feature': model.feature_names_in_,
                    'Importance': model.feature_importances_
                })

                def map_original_feature(feature_name):
                    if feature_name.startswith("Gender_"):
                        return "Gender"
                    elif feature_name.startswith("IncomeLevel_"):
                        return "IncomeLevel"
                    elif feature_name.startswith("Country_"):
                        return "Country"
                    else:
                        return feature_name

                importances['OriginalFeature'] = importances['Feature'].apply(map_original_feature)
                grouped_importances = importances.groupby('OriginalFeature').sum().reset_index()
                grouped_importances = grouped_importances.sort_values(by='Importance', ascending=False)

                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='OriginalFeature', data=grouped_importances, ax=ax)
                ax.set_title("Factors Influencing Prediction")
                st.pyplot(fig)

    elif app_mode == "Personalized Recommendations":
        st.header("🌍 Destination Recommendations")
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("Select Country", sorted(df['Country'].dropna().unique()))
        with col2:
            visit_mode = st.selectbox("Select Visit Mode", sorted(df['VisitMode'].dropna().unique()))

        if st.button("Get Recommendations", type="primary"):
            recommendations = get_recommendations(df, country=country, visit_mode=visit_mode)
            if recommendations.empty:
                st.warning(f"No attractions found for {visit_mode} trips in {country}")
            else:
                st.success(f"Top Attractions for {visit_mode} in {country}:")
                for i, (idx, row) in enumerate(recommendations.iterrows(), start=1):
                    st.markdown(f"**{i}. {row['Attraction']}** – ⭐ {row['Rating']:.2f}")

if __name__ == "__main__":
    main()