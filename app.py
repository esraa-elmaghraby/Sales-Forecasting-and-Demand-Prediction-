import streamlit as st
import pandas as pd
import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import plotly.graph_objects as go
import plotly.express as px

# Load model
model = pickle.load(open("model.pkl", "rb"))
@st.cache_data
def load_data():
    return pd.read_parquet("Data/train_reduced.parquet")

# Page settings
st.set_page_config(page_title="Sales Forecasting", layout="wide")

# Title
st.markdown("""
    <h2 style='text-align: center; color: white;'>
        TIME SERIES SALES FORECASTING
    </h2>
""", unsafe_allow_html=True)

# Styling
st.markdown("""
    <style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

.stApp {
        background-image: url("https://i.ibb.co/TMVJ4V1w/Wallpaper.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Roboto', sans-serif;
    }

    .stApp::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }

h1, h2, h3 {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
}
p {
    color: #E0E0E0 !important;  
}
div {
    color: #E0E0E0 !important;  
}

.stDateInput > div > div, .stSelectbox > div > div {
    background-color: rgba(255, 255, 255, 0.15);
    color: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
    border-radius: 5px;
    width: 600px !important; 
}
div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid #4A90E2 !important;
    border-radius: 5px !important;
    color: white !important;
    width: 600px !important; 
}
div[data-baseweb="select"] * {
    color: #E0E0E0 !important;
    background-color: transparent !important;
}
div[data-baseweb="popover"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #4A90E2 !important;
    border-radius: 5px;
}
div[data-baseweb="popover"] div[role="option"] {
    color: #FFFFFF !important;
    background-color: #000000 !important;
}
div[data-baseweb="popover"] div[role="option"]:hover {
    background-color: #4A90E2 !important;
}

div[data-baseweb="popover"] div[role="option"] {
    color: black !important;
    background-color: white !important;
}


div[data-testid="stDateInput"] {
    background-color: transparent !important;
    border: none !important;
    border-radius: 5px !important;
    color: #FFFFFF !important;
    width: 600px !important; 
}

div[data-testid="stDateInput"] input {
    color: #FFFFFF !important;
    background-color: transparent !important;
}
label {
    color: #E0E0E0 !important;
}
.stDateInput label, .stSelectbox label {
    color: #E0E0E0 !important;
}
.stCheckbox > label {
    color: #E0E0E0 !important;
    font-size: 16px !important; 
}
.stButton > button {
    background-color: #4A90E2;
    color: #FFFFFF !important;
    border: none;
    border-radius: 5px;
    padding: 8px 15px !important; 
    font-size: 14px !important;
    font-weight: bold;
    transition: background-color 0.3s;
    width: 300px !important; 
    text-align: center;
}
.stButton > button:hover {
    background-color: #357ABD;`
}
hr {
    border-top: 1px solid #4A90E2;
}
    .custom-frame {
        background-color: rgba(10, 12, 20, 0.8);  
        border: 1px solid #4A90E2;  
        border-radius: 10px;  
        padding: 15px;  
    }
    .custom-frame h3 {
        color: #FFFFFF !important;  
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        margin-bottom: 10px;
    }
    .custom-frame .metric-box {
        background-color: rgba(255, 255, 255, 0.1);  
        border: 1px solid #4A90E2;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        color: #FFFFFF !important;
    }
    .custom-frame .metric-label {
        font-size: 14px;
        color: #E0E0E0 !important;
    }
    .custom-frame .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF !important;
    }
    .custom-frame .insight {
        color: #E0E0E0 !important;
        margin-top: 10px;
    }
footer {
    color: #E0E0E0 !important;
    text-align: center;
    font-family: 'Roboto', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("TOTAL SALES", "23.8M")
with col2:
    st.metric("AVG DAILY SALES", "780,952")
with col3:
    st.metric("NUMBER OF ITEMS", "4018")
with col4:
    st.metric("Forecast Accuracy (R²)", "77%")

st.markdown("---")

# Input
train = load_data()
stores = pd.read_csv("Data/stores.csv")
items = pd.read_csv("Data/items.csv")
oil = pd.read_csv("Data/oil.csv")
holidays = pd.read_csv("Data/holidays_events.csv")
transactions = pd.read_csv("Data/transactions.csv")

oil['date'] = pd.to_datetime(oil['date'])
holidays['date'] = pd.to_datetime(holidays['date'])
transactions['date'] = pd.to_datetime(transactions['date'])

store_numbers = sorted(stores["store_nbr"].unique())
item_numbers = sorted(items["item_nbr"].unique())

def check_if_holiday_or_weekend(date, holidays):
    is_weekend = date.weekday() >= 5  # Saturday (5) or Sunday (6)
    is_holiday = not holidays[holidays['date'] == pd.to_datetime(date)].empty
    return is_weekend or is_holiday

def get_oil_price(date, oil):
    matched = oil[oil['date'] == pd.to_datetime(date)]
    if not matched.empty:
        return matched.iloc[0]['dcoilwtico']
    else:
        prev = oil[oil['date'] < pd.to_datetime(date)].sort_values(by='date', ascending=False)
        for _, row in prev.iterrows():
            if pd.notnull(row['dcoilwtico']):
                return row['dcoilwtico']
        return -1  

def get_store_and_item_info(store_nbr, item_nbr):
    store_info = stores[stores['store_nbr'] == store_nbr].iloc[0]
    item_info = items[items['item_nbr'] == item_nbr].iloc[0]
    type_ = store_info['type']
    cluster = store_info['cluster']
    family = item_info['family']
    return type_, cluster, family

def get_transactions(date, store_nbr, transactions):
    matched = transactions[(transactions['date'] == pd.to_datetime(date)) &
                               (transactions['store_nbr'] == store_nbr)]
    if not matched.empty:
        return matched.iloc[0]['transactions']
    else:
        return -1  

def get_perishable(item_nbr):
    matched = items[items['item_nbr'] == item_nbr]
    if not matched.empty:
        return int(matched.iloc[0]['perishable'])
    return 0 

def calculate_avg_sales(df,item_id=None, store_id=None):
        filtered_df = df.copy()
        if item_id is not None:
            filtered_df = filtered_df[filtered_df['item_nbr'] == item_id]
    
        if store_id is not None:
            filtered_df = filtered_df[filtered_df['store_nbr'] == store_id]
    
        return filtered_df['unit_sales'].mean()

def compute_time_dependent_features(date, train):
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    if date.year != 2017:
        return {
            'lag_7': -1,
            'rolling_mean_7': -1,
            'rolling_mean_30': -1
        }
    train['date'] = pd.to_datetime(train['date'])
    idx = train[train['date'] == pd.to_datetime(date)].index
    if len(idx) == 0:
        return {
            'lag_7': -1,
            'rolling_mean_7': -1,
            'rolling_mean_30': -1
        }
    idx = idx[0]
    try:
        lag_7 = train.loc[idx - 7, 'sales']
        rolling_mean_7 = train.loc[idx - 7:idx - 1, 'sales'].mean()
        rolling_mean_30 = train.loc[idx - 30:idx - 1, 'sales'].mean()
    except:
        return {
            'lag_7': -1,
            'rolling_mean_7': -1,
            'rolling_mean_30': -1
        }
    return {
        'lag_7': lag_7,
        'rolling_mean_7': rolling_mean_7,
        'rolling_mean_30': rolling_mean_30
    }

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Forecast Input")
    date = st.date_input("Date")
    store_nbr = st.selectbox("Store Number", store_numbers)
    item_nbr = st.selectbox("Item Number", item_numbers)
    
    col1, col2 = st.columns(2)
    with col1:
        onpromotion = st.checkbox("On Promotion?")
    
    if st.button("Predict Sales"):
        dayofweek = date.weekday()
        is_holiday = check_if_holiday_or_weekend(date, holidays)
        dcoilwtico = get_oil_price(date, oil)
        type_, cluster, family = get_store_and_item_info(store_nbr, item_nbr)
        transactions = get_transactions(date, store_nbr, transactions)
        perishable = get_perishable(item_nbr)
        selected_features = compute_time_dependent_features(date,train)
        lag_7 = selected_features['lag_7']
        rolling_mean_7 = selected_features['rolling_mean_7']
        rolling_mean_30 = selected_features['rolling_mean_30']


        model_input = {
            'store_nbr': store_nbr,
            'item_nbr': item_nbr,
            'onpromotion': int(onpromotion),
            'perishable': int(perishable),
            'lag_7': lag_7,
            'rolling_mean_7': rolling_mean_7,
            'rolling_mean_30': rolling_mean_30,
            'transactions': transactions,
            'dayofweek': dayofweek,
            'type': type_,
            'cluster': cluster,
            'dcoilwtico': dcoilwtico,
            'family': family,
            'is_holiday': int(is_holiday),
        }

        input_df = pd.DataFrame([model_input])

        input_df['type'] = LabelEncoder().fit_transform(input_df['type'])
        input_df['family'] = LabelEncoder().fit_transform(input_df['family'])

        prediction_log = model.predict(input_df)[0]
        predicted_sales = np.expm1(prediction_log)

    
        with right_col:
            avg_sales = calculate_avg_sales(train,item_nbr,store_nbr)
        
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=["Average Sales", "Predicted Sales"],
                y=[avg_sales, predicted_sales],
                mode='lines+markers',
                name='Sales',
                line=dict(color='#4A90E2', width=4),
                marker=dict(size=12, color='#000000', line=dict(color='#4A90E2', width=2))
            ))
        
            fig.update_layout(
                title=dict(
                    text="Predicted vs Average Item Sales",
                    font=dict(size=20, color='#4A90E2'),     
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Metric",
                yaxis_title="Sales Units",
                template='plotly_dark',
                height=220,
                plot_bgcolor='rgba(10, 12, 20, 0.8)',
                paper_bgcolor='rgba(10, 12, 20, 0.8)',
                font=dict(color='#FFFFFF', size=14),
                xaxis=dict(
                    tickfont=dict(color='#FFFFFF'),
                    gridcolor='rgba(255, 255, 255, 0.3)',
                    zerolinecolor='rgba(255, 255, 255, 0.4)'
                ),
                yaxis=dict(
                    tickfont=dict(color='#FFFFFF'),
                    gridcolor='rgba(255, 255, 255, 0.3)',
                    zerolinecolor='rgba(255, 255, 255, 0.4)'
                ),
                margin=dict(l=50, r=50, t=50, b=50),
            )
        
            if predicted_sales > avg_sales * 1.05:
                insight = "A significant increase in sales is forecasted."
            elif predicted_sales < avg_sales * 0.95:
                insight = "A noticeable drop in sales is expected."
            else:
                insight = "A stable sales trend is predicted."
        
            st.markdown(f"""
                <div style="background-color: rgba(10, 12, 20, 0.8); padding: 25px; border-radius: 0px;">
                    <h3 style="color: white;">Sales Prediction</h3>
                    <div style="border: 1px solid #555; border-radius: 0px; padding: 15px; margin-bottom: 25px;">
                        <div style="color: #BBBBBB; font-size: 16px;">Predicted Units</div>
                        <div style="color: #E0D6FF; font-size: 28px; font-weight: bold;">{predicted_sales:.2f} units</div>
                    </div>
            """, unsafe_allow_html=True)
        
            st.plotly_chart(fig, use_container_width=True)
        
            st.markdown(f"""
                    <p style="color: #BBBBBB; font-style: italic; margin-top: 15px; margin-bottom: 0;">{insight}</p>
                </div>
            """, unsafe_allow_html=True)


st.markdown("---")


# Footer
st.markdown("""
<div style="text-align: center;">
    Developed by <strong>DEPI Project Team</strong><br>
    Sales Forecasting & Demand Prediction Project <br>
    <a href="https://github.com/esraa-elmaghraby/Sales-Forecasting-and-Demand-Prediction-" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
