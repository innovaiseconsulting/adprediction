import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def create_streamlit_app() -> object:
    st.title('Advertising Prediction Analysis')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Select target variable
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        target_column = st.selectbox("Select the target variable", numeric_columns)

        # Select features
        feature_columns = st.multiselect("Select the feature variables",
                                         [col for col in numeric_columns if col != target_column],
                                         default=[col for col in numeric_columns if col != target_column])

        if len(feature_columns) > 0 and target_column:
            X = df[feature_columns]
            y = df[target_column]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Model performance
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("Model Performance:")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared Score: {r2:.2f}")

            # Display coefficients
            st.write("Model Coefficients:")
            coef_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient': model.coef_})
            st.write(coef_df)

            # Correlation heatmap
            st.write("Correlation Heatmap:")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[feature_columns + [target_column]].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Actual vs Predicted plot
            st.write("Actual vs Predicted Plot:")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel(f'Actual {target_column}')
            ax.set_ylabel(f'Predicted {target_column}')
            ax.set_title(f'Actual vs Predicted {target_column}')
            st.pyplot(fig)

            # Prediction interface
            st.write("Make a Prediction:")
            input_features = {}
            for feature in feature_columns:
                input_features[feature] = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))

            if st.button("Predict"):
                input_df = pd.DataFrame([input_features])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                st.write(f"Predicted {target_column}: {prediction:.2f}")


if __name__ == "__main__":
    create_streamlit_app()

