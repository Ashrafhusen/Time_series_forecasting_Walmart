import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from neuralprophet import NeuralProphet

# === App Configuration ===
st.set_page_config(page_title="🧠 Walmart Sales Forecast", layout="centered")

st.title("📈 Walmart Sales Forecasting App")
st.markdown("Powered by **NeuralProphet** and your trained model.")

# === Load the model ===
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model_path = "../output/prophet_model.pkl"  # Adjust path if needed
model = load_model(model_path)

# === User Input ===
periods = st.slider("🔮 How many future weeks to forecast?", min_value=5, max_value=52, value=20)

# === Load historical data used during training ===
@st.cache_data
def load_data():
    df = pd.read_csv("../input/train.csv")  # Original raw dataset
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Store'] == 1]  # You can make this dynamic later
    df = df.groupby("Date").agg({"Weekly_Sales": "sum"}).reset_index()
    df = df.rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    return df

df = load_data()

# === Forecast ===
future = model.make_future_dataframe(df, periods=periods)
forecast = model.predict(future)

# === Plotting ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Sales'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], mode='lines', name='Forecast'))
fig.update_layout(title="📊 Weekly Sales Forecast", xaxis_title="Date", yaxis_title="Sales ($)", height=500)
st.plotly_chart(fig, use_container_width=True)

# === Forecast Table ===
st.markdown("### 📋 Forecast Table")
st.dataframe(forecast[['ds', 'yhat1']].tail(periods).reset_index(drop=True))

# === Download Option ===
csv = forecast[['ds', 'yhat1']].tail(periods).to_csv(index=False)
st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
