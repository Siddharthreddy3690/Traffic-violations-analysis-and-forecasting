🚦 Traffic Violations Analysis and forecasting

An interactive dashboard built with Dash, Plotly, and Prophet to analyze and forecast traffic violations across Indian cities.

📌 Features

🔍 Overview Tab
KPIs: Total Violations, Total Fines, Penalty Points
Charts:
Violations by Type
Violations by City
Fine Amount by Violation Type
Penalty Points by Violation Type
Share of Violations (Pie)
Vehicle Share (Pie)

📈 Trend Analysis Tab
Choose Daily / Weekly / Monthly granularity
View violations trends over time
Track Fines & Penalty Points with rolling averages

⚖️ Comparative Analysis Tab
Grouped bar chart comparing violations by city & vehicle type
Box plot showing fine distribution by vehicle type

🤖 Predictive Analysis Tab
Forecast future violations using Prophet (with fallback to rolling mean if data sparse)
Forecast future fines based on predicted violations × average fine
Risk Indicator Gauge showing projected violation risk (Low/Medium/High)
Toggle between Forecast Charts and Forecast Data Table

🛠️ Tech Stack
Python 3.9+
Dash (UI framework)
Plotly (visualizations)
Prophet (time-series forecasting)
Pandas & NumPy (data wrangling)
Bootstrap (styling)

📂 Project Structure
traffic-voilation-analysis/
│
├── data/
│ └── Indian_Traffic_Violations.csv # Dataset
│
├── notebooks/
│ └── 01_Data_Cleaning.ipynb
│ └── 02_EDA.ipynb
│ └── 03_Visualization.ipynb
│ └── 04_Advanced_Analytics.ipynb
│
├── src/
│ └── utils.py
│
├── requirements.txt
├── README.md
└── report.md

📊 Steps in Analysis
1. Data Cleaning & Preparation
- Handle missing values (`Helmet_Worn`, `Seatbelt_Worn`, `Comments`)  
- Convert `Date` and `Time` into proper datetime  
- Normalize categorical values (`Yes/No` → consistent format)  
- Remove duplicates  

2. Exploratory Data Analysis (EDA)
- Most common violations  
- Fine distribution & statistics  
- State-wise violations  
- Driver demographics (age, gender)  
- Road & weather conditions impact  

3. Visualization & Insights
- Top 10 violation types (bar chart)  
- Violations over time (line chart)  
- Heatmap (day of week vs time of day)  
- Fine amount vs driver age (boxplot)  
- Payment methods (pie chart)  

4. Advanced Analytics
- Predict fine amounts (Regression)  
- Classify if **court appearance required** (Classification)  
- Cluster drivers into risk categories (Clustering)  

⚙️ Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/siddharthreddy3690/traffic-violations-analysis-and-forecasting.git
cd traffic-violations-dashboard

2️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

requirements.txt should contain:
dash
dash-bootstrap-components
plotly
prophet
pandas
numpy

4️⃣ Run the app
python app.py

5️⃣ Open in your browser:
http://127.0.0.1:8050

📊 Dataset
The dataset is expected to be stored in data/Indian_Traffic_Violations.csv
Required columns:
Date (datetime)
Violation_Type
Vehicle_Type
Location (City/Area)
Fine_Amount
Penalty_Points

🚀 Deployment
You can deploy the dashboard on:
Heroku
Render
Streamlit Cloud
Docker

📌 Future Improvements
Add geographical maps (city-level heatmaps)
Extend predictive models (XGBoost, LSTM for classification of risky cities/vehicles)
Real-time data ingestion
