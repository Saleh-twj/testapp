import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# 🎨 Enhanced Professional Styling
# ------------------------------
st.markdown(
    """
    <style>
    /* Main app background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #1e293b 75%, #0f172a 100%);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
        color: #ffffff;
        min-height: 100vh;
    }

    /* Enhanced content containers with professional glass effect */
    .block-container {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 10px;
    }

    /* Improve all Streamlit elements visibility */
    .stButton>button {
        background: linear-gradient(45deg, #2563eb, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stSelectbox, .stNumberInput, .stSlider {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Enhanced metric cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }

    /* Professional header */
    .main-header {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3) !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.3), rgba(29, 78, 216, 0.3)) !important;
        border-left: 4px solid #2563eb !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Custom header with animated gradient */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced metric cards with glass morphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        margin: 8px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.15);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #b0b0b0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Performance badges */
    .performance-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .excellent { background: linear-gradient(45deg, #00b09b, #96c93d); }
    .good { background: linear-gradient(45deg, #2193b0, #6dd5ed); }
    .fair { background: linear-gradient(45deg, #ff9a9e, #fecfef); }
    .poor { background: linear-gradient(45deg, #ff6b6b, #ffa8a8); }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #b0b0b0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
    }
    
    /* Progress and loading animations */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Custom section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stDownloadButton button {
        background: linear-gradient(45deg, #00b09b, #96c93d) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,176,155,0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# 🚀 Enhanced Header with Status
# ------------------------------
col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
with col_header1:
    st.markdown('<div class="main-header">📈 AI Stock Prediction Dashboard</div>', unsafe_allow_html=True)

with col_header2:
    st.metric("Status", "Ready", delta="Online")

with col_header3:
    st.metric("Last Update", datetime.now().strftime("%H:%M"))

# ------------------------------
# 🎯 Configuration Panel
# ------------------------------
st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)

config_col1, config_col2, config_col3, config_col4 = st.columns(4)

with config_col1:
    st.markdown("**📊 Data Parameters**")
    time_window = st.slider("Time Window (days)", 5, 100, 60, help="Number of days to look back for prediction")
    test_ratio = st.slider("Test Ratio", 0.1, 0.5, 0.2, help="Proportion of data for testing")

with config_col2:
    st.markdown("**🤖 Model Architecture**")
    model_choice = st.selectbox("Model Type", ["LSTM", "MLP", "Hybrid"], help="Choose neural network architecture")
    layers = st.slider("Hidden Layers", 1, 5, 3, help="Number of hidden layers")

with config_col3:
    st.markdown("**🔧 Training Parameters**")
    epochs = st.slider("Training Epochs", 10, 500, 100, help="Number of training iterations")
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1, help="Training batch size")

with config_col4:
    st.markdown("**📈 Prediction Settings**")
    forecast_days = st.slider("Forecast Days", 1, 30, 7, help="Days to forecast into future")
    confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, help="Prediction confidence interval")

# ------------------------------
# 🏢 TASI Stock Search & Data Management
# ------------------------------
st.markdown('<div class="section-header">📁 Data Management</div>', unsafe_allow_html=True)

# TASI companies for historical data
tasi_stocks = {
    "Al Rajhi Bank": "1120.SR",
    "SABIC": "2010.SR", 
    "Saudi Aramco": "2222.SR",
    "Alinma Bank": "1150.SR",
    "STC": "7010.SR",
    "Riyad Bank": "1010.SR",
    "Saudi British Bank": "1060.SR",
    "Arab National Bank": "1080.SR",
    "Saudi Electricity": "5110.SR",
    "Mobily": "7020.SR",
    "Zain KSA": "7030.SR",
    "Almarai": "2280.SR",
    "Savola Group": "2050.SR",
    "Jarir Marketing": "4190.SR",
    "Saudi Cement": "3030.SR",
    "Yamama Cement": "3020.SR"
}

# Data source selection
data_source = st.radio(
    "**Select Data Source:**",
    ["📤 Upload File", "🏢 TASI Stock Search"],
    horizontal=True
)

df = None

if data_source == "🏢 TASI Stock Search":
    col_search1, col_search2, col_search3 = st.columns([2, 1, 1])
    
    with col_search1:
        selected_stock = st.selectbox(
            "🔍 Select TASI Company", 
            options=list(tasi_stocks.keys()),
            help="Choose a company from TASI to load historical stock data"
        )
    
    with col_search2:
        period = st.selectbox(
            "📅 Historical Period",
            ["3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
    
    with col_search3:
        st.markdown("###")  # Vertical spacing
        load_data = st.button("📥 Load Stock Data", use_container_width=True)
    
    if load_data:
        with st.spinner(f"📊 Loading historical data for {selected_stock}..."):
            try:
                symbol = tasi_stocks[selected_stock]
                stock_data = yf.download(symbol, period=period)
                
                if not stock_data.empty:
                    df = stock_data.reset_index()
                    # Ensure we have the required columns
                    if 'Close' not in df.columns:
                        st.error("❌ No closing price data available for this stock")
                    else:
                        st.success(f"✅ Loaded {len(df)} days of historical data for {selected_stock}")
                        st.session_state.df = df
                else:
                    st.error("❌ No historical data found for this stock")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")

else:  # File Upload
    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        uploaded_file = st.file_uploader("**Upload Market Data**", type=["csv", "xlsx"], 
                                       help="Upload CSV or Excel file with OHLC data")

    with upload_col2:
        st.markdown("**Sample Data**")
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            # Generate sample data
            dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
            sample_data = pd.DataFrame({
                'Date': dates,
                'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
                'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
                'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            })
            sample_data['Close'] = sample_data['Close'].abs()  # Ensure positive prices
            csv = sample_data.to_csv(index=False)
            st.download_button("📥 Download Sample", data=csv, file_name="sample_stock_data.csv", mime="text/csv")

    if uploaded_file is not None:
        if uploaded_file.name.endswith("xlsx"):
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("📑 Select Worksheet", xls.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_csv(uploaded_file)

# Check if we have data from either source
if df is not None and 'Close' in df.columns:
    # Data preview with enhanced styling
    st.success(f"✅ Successfully loaded data with {len(df)} records")
    
    preview_col1, preview_col2 = st.columns([3, 1])
    with preview_col1:
        st.markdown("**🔍 Data Preview**")
        st.dataframe(df.head(10), use_container_width=True)
    
    with preview_col2:
        st.markdown("**📋 Data Summary**")
        st.metric("Total Records", len(df))
        if 'Date' in df.columns:
            date_range = f"{df['Date'].min()} to {df['Date'].max()}"
        else:
            date_range = f"Index {df.index.min()} to {df.index.max()}"
        st.metric("Date Range", date_range)
        try:
            price_change = float(((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100)
            st.metric("Total Return", f"{price_change:.2f}%")
        except:
            st.metric("Total Return", "N/A")

    # ------------------------------
    # 🔄 DATA NORMALIZATION - FIXED VERSION
    # ------------------------------
    st.markdown('<div class="section-header">🔄 Data Preprocessing</div>', unsafe_allow_html=True)
    
    def normalize_stock_data(df):
        """
        Normalize stock data by cleaning and applying MinMax scaling
        """
        df = df.copy()
        
        st.info("🔄 Cleaning and normalizing stock data...")
        
        # Determine which columns we have available
        available_columns = [str(col) for col in df.columns.tolist()]
        st.write(f"📊 Available columns: {', '.join(available_columns)}")
        
        # Define target columns for normalization (EXCLUDE 'Close' for model training)
        possible_columns = ["Price", "Open", "High", "Low", "Vol.", "Change %", "Volume"]
        columns_to_normalize = []
        
        for col in possible_columns:
            if col in df.columns:
                columns_to_normalize.append(col)
        
        st.write(f"🎯 Columns to normalize: {', '.join(columns_to_normalize)}")
        
        # Clean Volume/Vol. column if it exists - FIXED VERSION
        volume_column = None
        if "Vol." in df.columns:
            volume_column = "Vol."
        elif "Volume" in df.columns:
            volume_column = "Volume"
        
        if volume_column:
            def clean_volume(val):
                try:
                    # Handle NaN/None values first
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return 0.0
                    
                    # Handle string values
                    if isinstance(val, str):
                        val = val.replace(",", "").strip()
                        if val.endswith("B"):
                            return float(val[:-1]) * 1_000_000_000
                        elif val.endswith("M"):
                            return float(val[:-1]) * 1_000_000
                        elif val.endswith("K"):
                            return float(val[:-1]) * 1_000
                        else:
                            return float(val)
                    
                    # Handle numeric values
                    return float(val)
                    
                except (ValueError, TypeError):
                    return 0.0
            
            # Apply cleaning safely using list comprehension
            df[volume_column] = [clean_volume(x) for x in df[volume_column]]
            st.success(f"✅ Cleaned {volume_column} column")
        
        # Clean Change % column if it exists
        if "Change %" in df.columns:
            try:
                df["Change %"] = df["Change %"].astype(str).str.replace("%", "", regex=False)
                df["Change %"] = pd.to_numeric(df["Change %"], errors='coerce').fillna(0)
                st.success("✅ Cleaned Change % column")
            except Exception as e:
                st.warning(f"⚠️ Could not clean Change % column: {e}")
        
        # Clean numeric columns (remove commas and convert to float)
        numeric_columns = ["Price", "Open", "High", "Low"]
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(str).str.replace(",", "")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    st.success(f"✅ Cleaned {col} column")
                except Exception as e:
                    st.warning(f"⚠️ Could not clean {col} column: {e}")
        
        # Clean Close column separately (don't normalize it for model training)
        if "Close" in df.columns:
            try:
                df["Close"] = df["Close"].astype(str).str.replace(",", "")
                df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
                # Remove rows where Close is NaN after conversion
                df = df.dropna(subset=['Close'])
                st.success("✅ Cleaned Close column")
            except Exception as e:
                st.error(f"❌ Error cleaning Close column: {e}")
        
        # Apply MinMax scaling to selected columns (EXCLUDING 'Close')
        if columns_to_normalize:
            scaler = MinMaxScaler()
            
            # Only normalize columns that have numeric data
            valid_columns = []
            for col in columns_to_normalize:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    valid_columns.append(col)
            
            if valid_columns:
                try:
                    # Handle any remaining NaN values
                    df[valid_columns] = df[valid_columns].fillna(0)
                    
                    # Apply normalization
                    df[valid_columns] = scaler.fit_transform(df[valid_columns])
                    st.success(f"✅ Applied MinMax normalization to: {', '.join(valid_columns)}")
                    
                    # Show normalization examples
                    if len(valid_columns) > 0:
                        st.markdown("**🔍 Normalization Examples:**")
                        norm_cols = st.columns(min(3, len(valid_columns)))
                        for i, col in enumerate(valid_columns[:3]):
                            with norm_cols[i]:
                                st.metric(f"{col} (Normalized)", f"{df[col].iloc[0]:.3f}")
                                
                except Exception as e:
                    st.error(f"❌ Error during normalization: {e}")
            else:
                st.warning("⚠️ No valid numeric columns found for normalization")
        else:
            st.warning("⚠️ No columns available for normalization")
        
        return df

    # Apply normalization
    with st.spinner("🔄 Normalizing stock data..."):
        try:
            df_normalized = normalize_stock_data(df)
            df = df_normalized
            st.success("✅ Data normalization completed!")
            
            # Show normalized data preview
            col_norm1, col_norm2 = st.columns([2, 1])
            
            with col_norm1:
                st.markdown("**📊 Normalized Data Preview**")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col_norm2:
                st.markdown("**📈 Data Ranges**")
                # Show Close price range (should be original values)
                if 'Close' in df.columns:
                    st.metric(
                        "Close Price Range", 
                        f"${df['Close'].min():.2f} - ${df['Close'].max():.2f}"
                    )
                
                # Show normalized columns ranges
                normalized_cols = [col for col in ["Open", "High", "Low", "Vol.", "Volume", "Change %"] 
                                 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                
                for col in normalized_cols[:3]:
                    if col in df.columns:
                        st.metric(
                            f"{col} Range", 
                            f"{df[col].min():.3f} - {df[col].max():.3f}"
                        )
                        
        except Exception as e:
            st.error(f"❌ Normalization failed: {e}")
            st.info("⚠️ Continuing with original data...")

    # ------------------------------
    # 🤖 Model Training Section
    # ------------------------------
    st.markdown('<div class="section-header">🤖 AI Model Training</div>', unsafe_allow_html=True)
    
    with st.spinner("🚀 Training AI model... This may take a few moments"):
        # Preprocessing
        close_prices = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_prices)

        X, y = [], []
        for i in range(time_window, len(scaled)):
            X.append(scaled[i - time_window:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)

        if model_choice == "LSTM":
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build enhanced model
        model = Sequential()
        if model_choice == "MLP":
            model.add(Dense(256, activation="relu", input_shape=(X_train.shape[1],)))
            model.add(Dropout(0.3))
            for i in range(layers - 1):
                model.add(Dense(128 // (i + 1), activation="relu"))
                model.add(Dropout(0.2))
            model.add(Dense(1))
        elif model_choice == "LSTM":
            model.add(LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.3))
            for i in range(layers - 1):
                model.add(LSTM(100 // (i + 1), return_sequences=(i < layers - 2)))
                model.add(Dropout(0.2))
            model.add(Dense(1))
        else:  # Hybrid
            model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.3))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse", metrics=['mae'])

        # Training with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        progress_bar.progress(100)
        status_text.success("✅ Model training completed successfully!")

    # ------------------------------
    # 📊 Predictions & Evaluation
    # ------------------------------
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)

    # ------------------------------
    # 🎯 Performance Metrics
    # ------------------------------
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
    
    # Performance badges
    def get_performance_badge(metric, value):
        if metric == "R²":
            if value >= 0.9: return "excellent"
            elif value >= 0.7: return "good"
            elif value >= 0.5: return "fair"
            else: return "poor"
        elif metric == "RMSE":
            avg_price = np.mean(y_test_inv)
            relative_error = value / avg_price
            if relative_error < 0.02: return "excellent"
            elif relative_error < 0.05: return "good"
            elif relative_error < 0.1: return "fair"
            else: return "poor"
        return "fair"

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rmse_badge = get_performance_badge("RMSE", rmse)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Root Mean Square Error</div>
            <div class="metric-value" style="color: #ff6b6b;">{rmse:.2f}</div>
            <span class="performance-badge {rmse_badge}">{rmse_badge.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Square Error</div>
            <div class="metric-value" style="color: #feca57;">{mse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        r2_badge = get_performance_badge("R²", r2)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R² Score</div>
            <div class="metric-value" style="color: #1dd1a1;">{r2:.4f}</div>
            <span class="performance-badge {r2_badge}">{r2_badge.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Absolute % Error</div>
            <div class="metric-value" style="color: #ff9ff3;">{mape:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------
    # 📈 Enhanced Visualization Tabs
    # ------------------------------
    st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Price Prediction", "🕯️ Market Analysis", "📉 Loss Metrics", "🔮 Future Forecast"])

    with tabs[0]:
        col_chart1, col_chart2 = st.columns([3, 1])
        with col_chart1:
            # Enhanced line chart with confidence interval
            fig = go.Figure()
            
            # Actual prices
            fig.add_trace(go.Scatter(
                y=y_test_inv.flatten(),
                name="Actual Prices",
                line=dict(color="#1dd1a1", width=3),
                mode='lines'
            ))
            
            # Predicted prices
            fig.add_trace(go.Scatter(
                y=y_pred_inv.flatten(),
                name="Predicted Prices",
                line=dict(color="#ff6b6b", width=3, dash='dash'),
                mode='lines'
            ))
            
            fig.update_layout(
                title="Actual vs Predicted Stock Prices",
                xaxis_title="Time Period",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.markdown("**📈 Prediction Accuracy**")
            accuracy = max(0, (1 - mape/100) * 100)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = accuracy,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Accuracy"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 107, 107, 0.2)"},
                        {'range': [50, 80], 'color': "rgba(254, 202, 87, 0.2)"},
                        {'range': [80, 100], 'color': "rgba(29, 209, 161, 0.2)"}],
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tabs[1]:
        # Enhanced candlestick chart
        st.subheader("Market Analysis - Candlestick Chart")
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            candlestick = go.Figure(data=[go.Candlestick(
                x=df.index[-100:],
                open=df["Open"].tail(100),
                high=df["High"].tail(100),
                low=df["Low"].tail(100),
                close=df["Close"].tail(100),
                increasing_line_color='#1dd1a1',
                decreasing_line_color='#ff6b6b'
            )])
            candlestick.update_layout(
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=500,
                title="Last 100 Trading Days"
            )
            st.plotly_chart(candlestick, use_container_width=True)
        else:
            st.warning("⚠️ OHLC data required for candlestick chart")

    with tabs[2]:
        # Enhanced training history
        col_loss1, col_loss2 = st.columns([3, 1])
        with col_loss1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Loss plot
            ax1.plot(history.history["loss"], label="Training Loss", color="#667eea", linewidth=2)
            ax1.plot(history.history["val_loss"], label="Validation Loss", color="#ff6b6b", linewidth=2)
            ax1.set_title("Model Training History", fontsize=14, fontweight='bold', color='white')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#0c0c0c')
            
            # MAE plot
            if 'mae' in history.history:
                ax2.plot(history.history["mae"], label="Training MAE", color="#1dd1a1", linewidth=2)
                ax2.plot(history.history["val_mae"], label="Validation MAE", color="#feca57", linewidth=2)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#0c0c0c')
            
            fig.patch.set_facecolor('#0c0c0c')
            st.pyplot(fig)
        
        with col_loss2:
            st.markdown("**📋 Training Summary**")
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            st.metric("Final Train Loss", f"{final_loss:.4f}")
            st.metric("Final Val Loss", f"{final_val_loss:.4f}")
            st.metric("Training Epochs", len(history.history['loss']))

    with tabs[3]:
        st.subheader("Future Price Forecast")
        # Simple future prediction visualization
        last_sequence = scaled[-time_window:]
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            next_pred = model.predict(current_sequence.reshape(1, time_window, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]
        
        future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Create future dates - FIXED VERSION
        if hasattr(df.index, 'dtype'):
            if 'datetime' in str(df.index.dtype).lower():
                last_date = df.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            else:
                # If index is not datetime, create sequential future indices
                future_dates = list(range(len(df), len(df) + forecast_days))
        else:
            # Fallback: create simple numeric indices
            future_dates = list(range(len(df), len(df) + forecast_days))
        
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices.flatten(),
            name="Forecast",
            line=dict(color="#667eea", width=3),
            mode='lines+markers'
        ))
        
        fig_future.update_layout(
            title=f"{forecast_days}-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Price ($)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_future, use_container_width=True)

    # ------------------------------
    # 💾 Export Results
    # ------------------------------
    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
    
    pred_df = pd.DataFrame({
        "Actual_Price": y_test_inv.flatten(),
        "Predicted_Price": y_pred_inv.flatten(),
        "Absolute_Error": np.abs(y_test_inv.flatten() - y_pred_inv.flatten()),
        "Percentage_Error": (np.abs(y_test_inv.flatten() - y_pred_inv.flatten()) / y_test_inv.flatten()) * 100
    })
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        csv = pred_df.to_csv(index=False)
        st.download_button(
            "📥 Download Predictions", 
            data=csv, 
            file_name="stock_predictions.csv", 
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        # Model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary_text = "\n".join(model_summary)
        st.download_button(
            "📋 Model Architecture", 
            data=model_summary_text, 
            file_name="model_architecture.txt", 
            mime="text/plain",
            use_container_width=True
        )
    
    with col_export3:
        # Training report
        report = f"""
        Stock Prediction Model Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Model Type: {model_choice}
        Time Window: {time_window} days
        Test Ratio: {test_ratio}
        Training Epochs: {epochs}
        Batch Size: {batch_size}
        
        Performance Metrics:
        - RMSE: {rmse:.4f}
        - MSE: {mse:.4f}
        - R² Score: {r2:.4f}
        - MAPE: {mape:.2f}%
        
        Dataset: {uploaded_file.name if data_source == '📤 Upload File' else selected_stock}
        Records: {len(df)}
        """
        st.download_button(
            "📄 Training Report", 
            data=report, 
            file_name="training_report.txt", 
            mime="text/plain",
            use_container_width=True
        )

else:
    # Welcome state with sample visualization
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.05); border-radius: 20px; margin: 2rem 0;'>
        <h2 style='color: #667eea; margin-bottom: 1rem;'>🚀 Welcome to AI Stock Predictor</h2>
        <p style='color: #b0b0b0; font-size: 1.2rem; max-width: 600px; margin: 0 auto;'>
            Choose a data source above to get started with AI-powered stock predictions.
            Load TASI stock data automatically or upload your own historical market data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample visualization
    st.markdown("### 📊 How it works:")
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🏢</div>
            <h4>Select Stock</h4>
            <p style='color: #b0b0b0;'>Choose from TASI companies or upload your data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_demo2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
            <h4>AI Training</h4>
            <p style='color: #b0b0b0;'>Neural networks learn market patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_demo3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🔮</div>
            <h4>Get Predictions</h4>
            <p style='color: #b0b0b0;'>Receive accurate price forecasts</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# 📱 Footer
# ------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "© 2024 AI Stock Prediction Dashboard | Built with Streamlit & TensorFlow"
        "</div>", 
        unsafe_allow_html=True
    )