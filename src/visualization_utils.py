# src/visualization_utils.py
"""
Visualization helper functions for Traffic Violations Analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_violations(df, n=10):
    violation_counts = df['Violation_Type'].value_counts().head(n)
    plt.figure(figsize=(10,6))
    sns.barplot(x=violation_counts.values, y=violation_counts.index, palette="viridis")
    plt.title(f"Top {n} Traffic Violations")
    plt.xlabel("Count")
    plt.ylabel("Violation Type")
    plt.show()

def plot_fine_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Fine_Amount'], bins=30, kde=True)
    plt.title("Distribution of Fine Amounts")
    plt.xlabel("Fine Amount")
    plt.ylabel("Frequency")
    plt.show()

def plot_statewise_violations(df, n=10):
    state_counts = df['Location'].value_counts().head(n)
    plt.figure(figsize=(10,6))
    sns.barplot(x=state_counts.values, y=state_counts.index, palette="magma")
    plt.title(f"Top {n} States by Violations")
    plt.xlabel("Count")
    plt.ylabel("State")
    plt.show()

def plot_gender_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Driver_Gender", palette="coolwarm")
    plt.title("Violations by Gender")
    plt.show()

def plot_age_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Driver_Age'], bins=20, kde=True)
    plt.title("Distribution of Driver Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

def plot_weather_conditions(df):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Weather_Condition", order=df['Weather_Condition'].value_counts().index, palette="Set2")
    plt.title("Violations by Weather Condition")
    plt.xticks(rotation=45)
    plt.show()

def plot_road_conditions(df):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Road_Condition", order=df['Road_Condition'].value_counts().index, palette="Set3")
    plt.title("Violations by Road Condition")
    plt.xticks(rotation=45)
    plt.show()
