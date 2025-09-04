# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("D:\\Learning\\DataSet\\Salary.csv")

# Remove outliers
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1
Upper_limit = Q3 + (1.5 * IQR)
Lower_limit = Q1 - (1.5 * IQR)
df = df[(df["salary"] < Upper_limit) & (df["salary"] > Lower_limit)]

# Fill missing values
df["education_level"].fillna(df["education_level"].mode()[0], inplace=True)
df["job_role"].fillna(df["job_role"].mode()[0], inplace=True)
df["industry"].fillna(df["industry"].mode()[0], inplace=True)
df["performance_score"].fillna(df["performance_score"].mean(), inplace=True)

# Define features
target = "salary"
numerical_features = ["age", "years_experience", "certifications", "hours_per_week",
                      "performance_score", "communication_skill", "leadership_skill"]
categorical_features = ["education_level", "job_role", "industry", "location"]

# Split data
X = df[numerical_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# # Save the pipeline
# joblib.dump(pipeline, "model_pipeline.pkl")
# print("✅ Model trained and saved as 'model_pipeline.pkl'")

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred)
print(r2)

