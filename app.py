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
# ---------- LANGUAGES ----------
# ------------------------------
LANG = {
    "English": {
        "page_title": "Stock Prediction Center",
        "status": "Status",
        "last_update": "Last Update",
        "model_config": "⚙️ Model Configuration",
        "data_parameters": "📊 Data Parameters",
        "time_window": "Time Window (days)",
        "test_ratio": "Test Ratio",
        "model_architecture": "🤖 Model Architecture",
        "model_type": "Model Type",
        "hidden_layers": "Hidden Layers",
        "training_params": "🔧 Training Parameters",
        "epochs": "Training Epochs",
        "batch_size": "Batch Size",
        "prediction_settings": "📈 Prediction Settings",
        "forecast_days": "Forecast Days",
        "confidence_level": "Confidence Level",
        "data_management": "📁 Data Management",
        "select_data_source": "Select Data Source:",
        "upload_file": "📤 Upload File",
        "tasi_search": "🏢 TASI Stock Search",
        "load_stock_data": "📥 Load Stock Data",
        "historical_period": "📅 Historical Period",
        "data_preview": "🔍 Data Preview",
        "data_summary": "📋 Data Summary",
        "total_records": "Total Records",
        "date_range": "Date Range",
        "total_return": "Total Return",
        "data_preprocessing": "🔄 Data Preprocessing",
        "available_columns": "Available columns:",
        "columns_to_normalize": "Columns to normalize:",
        "cleaned_column": "Cleaned {col} column",
        "applied_normalization": "Applied MinMax normalization to:",
        "no_numeric_cols": "No valid numeric columns found for normalization",
        "no_cols_available": "No columns available for normalization",
        "normalization_completed": "Data normalization completed!",
        "ai_model_training": "🤖 AI Model Training",
        "training_completed": "Model training completed successfully!",
        "model_performance": "📊 Model Performance",
        "root_mean_square_error": "Root Mean Square Error",
        "mean_square_error": "Mean Square Error",
        "r2_score": "R² Score",
        "mean_absolute_percentage_error": "Mean Absolute % Error",
        "advanced_analytics": "📈 Advanced Analytics",
        "price_prediction": "📊 Price Prediction",
        "market_analysis": "🕯️ Market Analysis",
        "loss_metrics": "📉 Loss Metrics",
        "future_forecast_tab": "🔮 Future Forecast",
        "prediction_accuracy": "Prediction Accuracy",
        "market_analysis_candlestick": "Market Analysis - Candlestick Chart",
        "training_summary": "📋 Training Summary",
        "future_price_forecast": "Future Price Forecast",
        "export_results": "💾 Export Results",
        "download_predictions": "📥 Download Predictions",
        "model_architecture_file": "📋 Model Architecture",
        "training_report": "📄 Training Report",
        "welcome_title": "🚀 Welcome to AI Stock Predictor",
        "welcome_text": "Choose a data source above to get started with AI-powered stock predictions. Load TASI stock data automatically or upload your own historical market data.",
        "how_it_works": "📊 How it works:",
        "select_stock_step": "Select Stock",
        "ai_training_step": "AI Training",
        "get_predictions_step": "Get Predictions",
        "learn_more": "Learn More",
        "learn_more_text": """
### About this Application
This system predicts future stock prices for companies listed on the Saudi Stock Exchange (TASI).
It uses **historical data** (downloaded from Yahoo Finance) and applies neural networks (LSTM / MLP / Hybrid)
to forecast short-term movements.

#### How to use:
1. Choose your data source: upload a CSV/XLSX with OHLC data or pick a TASI stock from the list.
2. Configure the model parameters (time window, model type, epochs, etc.) inside the Model Configuration expander.
3. Load the stock / upload the file, then allow normalization to run.
4. Train the model and view predictions, charts and performance metrics.
5. Export predictions or model architecture using the download buttons.

#### Notes:
- Predictions are experimental and for educational purposes only.
- Ensure your uploaded file has a 'Close' column and preferably 'Date' for time-based charts.
""",
        "footer_text": "© 2024 Stock Prediction Center | Built with Streamlit & TensorFlow",
        "select_lang": "Language",
        "sample_data": "🎲 Generate Sample Data",
        "download_sample": "📥 Download Sample",
        "generate_sample_text": "Generate sample OHLC data for testing",
        "no_data_loaded": "No data loaded yet — please choose a data source above.",
        "error_loading": "Error loading data:",
        "no_close": "No closing price data available for this stock",
        "no_historical": "No historical data found for this stock",
        "cleaning_info": "Cleaning and normalizing stock data...",
        "cleaning_warning": "Could not clean {col} column:",
        "normalization_examples": "🔍 Normalization Examples:",
        "close_range": "Close Price Range",
        "range": "Range",
        "download_excel": "Download Predictions as Excel"
    },
    "العربية": {
        "page_title": "مركز توقعات الأسهم",
        "status": "الحالة",
        "last_update": "آخر تحديث",
        "model_config": "⚙️ إعدادات النموذج",
        "data_parameters": "📊 معلمات البيانات",
        "time_window": "نافذة الزمن (أيام)",
        "test_ratio": "نسبة الاختبار",
        "model_architecture": "🤖 بنية النموذج",
        "model_type": "نوع النموذج",
        "hidden_layers": "الطبقات المخفية",
        "training_params": "🔧 معلمات التدريب",
        "epochs": "تكرارات التدريب",
        "batch_size": "حجم الدفعة",
        "prediction_settings": "📈 إعدادات التنبؤ",
        "forecast_days": "أيام التنبؤ",
        "confidence_level": "مستوى الثقة",
        "data_management": "📁 إدارة البيانات",
        "select_data_source": "اختر مصدر البيانات:",
        "upload_file": "📤 رفع ملف",
        "tasi_search": "🏢 بحث عن أسهم تاسي",
        "load_stock_data": "📥 تحميل بيانات السهم",
        "historical_period": "📅 الفترة التاريخية",
        "data_preview": "🔍 معاينة البيانات",
        "data_summary": "📋 ملخص البيانات",
        "total_records": "إجمالي السجلات",
        "date_range": "نطاق التواريخ",
        "total_return": "العائد الكلي",
        "data_preprocessing": "🔄 معالجة البيانات",
        "available_columns": "الأعمدة المتاحة:",
        "columns_to_normalize": "الأعمدة المطلوب تطبيعها:",
        "cleaned_column": "تم تنظيف عمود {col}",
        "applied_normalization": "تم تطبيق تطبيع MinMax على:",
        "no_numeric_cols": "لا توجد أعمدة رقمية صالحة للتطبيع",
        "no_cols_available": "لا توجد أعمدة متاحة للتطبيع",
        "normalization_completed": "اكتمل تطبيع البيانات!",
        "ai_model_training": "🤖 تدريب نموذج الذكاء الاصطناعي",
        "training_completed": "تم تدريب النموذج بنجاح!",
        "model_performance": "📊 أداء النموذج",
        "root_mean_square_error": "الجذر التربيعي لمتوسط الخطأ",
        "mean_square_error": "متوسط مربع الخطأ",
        "r2_score": "معامل التحديد R²",
        "mean_absolute_percentage_error": "متوسط الخطأ المطلق بالنسبة المئوية",
        "advanced_analytics": "📈 تحليلات متقدمة",
        "price_prediction": "📊 توقع السعر",
        "market_analysis": "🕯️ تحليل السوق",
        "loss_metrics": "📉 مقاييس الخسارة",
        "future_forecast_tab": "🔮 التنبؤ المستقبلي",
        "prediction_accuracy": "دقة التنبؤ",
        "market_analysis_candlestick": "تحليل السوق - مخطط الشموع",
        "training_summary": "📋 ملخص التدريب",
        "future_price_forecast": "توقع الأسعار المستقبلية",
        "export_results": "💾 تصدير النتائج",
        "download_predictions": "📥 تحميل التنبؤات",
        "model_architecture_file": "📋 بنية النموذج",
        "training_report": "📄 تقرير التدريب",
        "welcome_title": "🚀 مرحباً بك في متنبئ الأسهم بالذكاء الاصطناعي",
        "welcome_text": "اختر مصدر بيانات أعلاه للبدء بتوقعات أسعار الأسهم باستخدام الذكاء الاصطناعي. يمكنك تحميل بياناتك أو اختيار سهم من تاسي.",
        "how_it_works": "📊 كيف يعمل:",
        "select_stock_step": "اختر السهم",
        "ai_training_step": "تدريب الذكاء الاصطناعي",
        "get_predictions_step": "الحصول على التنبؤات",
        "learn_more": "معلومات تفصيلية",
        "learn_more_text": """
### عن هذا النظام
يقوم هذا النظام بالتنبؤ بأسعار الأسهم المستقبلية للشركات المدرجة في السوق المالية السعودية (تاسي).
يعتمد على **البيانات التاريخية** من Yahoo Finance ويستخدم شبكات عصبونية (مثل LSTM) لتوقع الحركات قصيرة المدى.

#### كيفية الاستخدام:
1. اختر مصدر البيانات: ارفع ملف CSV/XLSX يحتوي أعمدة OHLC أو اختر سهم من قائمة تاسي.
2. اضبط إعدادات النموذج داخل إطار إعدادات النموذج (مثل نافذة الزمن، نوع النموذج، عدد التكرارات).
3. اضغط على زر تحميل البيانات ثم انتظر عملية التنظيف والتطبيع.
4. درب النموذج لمشاهدة التنبؤات والمخططات ومقاييس الأداء.
5. يمكنك تنزيل التنبؤات أو تقرير التدريب أو بنية النموذج.

#### ملاحظات:
- التنبؤات تجريبية ولأغراض تعليمية فقط.
- تأكد من أن الملف المرفوع يحتوي عمود 'Close' للتشغيل الصحيح.
        """,
        "footer_text": "© 2024 مركز توقعات الأسهم | مبني باستخدام Streamlit و TensorFlow",
        "select_lang": "اللغة",
        "sample_data": "🎲 توليد بيانات تجريبية",
        "download_sample": "📥 تحميل نسخة تجريبية",
        "generate_sample_text": "توليد بيانات OHLC للاختبار",
        "no_data_loaded": "لا توجد بيانات محملة - الرجاء اختيار مصدر بيانات أعلاه.",
        "error_loading": "خطأ في تحميل البيانات:",
        "no_close": "لا توجد بيانات سعر إغلاق لهذا السهم",
        "no_historical": "لا توجد بيانات تاريخية لهذا السهم",
        "cleaning_info": "جاري تنظيف وتطبيع بيانات السهم...",
        "cleaning_warning": "تعذر تنظيف عمود {col}:",
        "normalization_examples": "🔍 أمثلة على التطبيع:",
        "close_range": "نطاق سعر الإغلاق",
        "range": "النطاق",
        "download_excel": "تحميل التنبؤات كملف Excel"
    }
}

# ------------------------------
# Page config & styling
# ------------------------------
st.set_page_config(page_title="Stock Prediction Center", layout="wide")

# Language selector in sidebar
lang_choice = st.sidebar.selectbox(LANG["English"]["select_lang"] if "select_lang" in LANG["English"] else "Language", list(LANG.keys()))
L = LANG[lang_choice]

# ------------------------------
# 🎨 Custom CSS (kept from your original)
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
    .block-container {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 10px;
    }
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
    .metric-card {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    .main-header {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3) !important;
    }
    .section-header {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.3), rgba(29, 78, 216, 0.3)) !important;
        border-left: 4px solid #2563eb !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
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
    .metric-title { font-size: 0.9rem; color: #b0b0b0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .performance-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-left: 8px; }
    .excellent { background: linear-gradient(45deg, #00b09b, #96c93d); }
    .good { background: linear-gradient(45deg, #2193b0, #6dd5ed); }
    .fair { background: linear-gradient(45deg, #ff9a9e, #fecfef); }
    .poor { background: linear-gradient(45deg, #ff6b6b, #ffa8a8); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: rgba(255, 255, 255, 0.1); border-radius: 12px 12px 0 0; padding: 12px 24px; border: 1px solid rgba(255,255,255,0.1); color: #b0b0b0; font-weight: 500; transition: all 0.3s ease; }
    .stTabs [aria-selected="true"] { background: linear-gradient(45deg, #667eea, #764ba2) !important; color: white !important; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
    .uploadedFile { background: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 12px !important; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
    .section-header { background: linear-gradient(90deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)); padding: 1rem 1.5rem; border-radius: 12px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #667eea; }
    .stDownloadButton button { background: linear-gradient(45deg, #00b09b, #96c93d) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 12px 24px !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
    .stDownloadButton button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,176,155,0.4); }
    .inline-gradient-title { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; font-weight: 800; margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# 🚀 Header
# ------------------------------
col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
with col_header1:
    st.markdown(f'<div class="inline-gradient-title">📈 {L["page_title"]}</div>', unsafe_allow_html=True)

with col_header2:
    st.metric(L["status"], "Ready", delta="Online")

with col_header3:
    st.metric(L["last_update"], datetime.now().strftime("%H:%M"))

# ------------------------------
# ⚙️ Configuration Panel
# ------------------------------
with st.expander(L["model_config"], expanded=False):
    st.markdown(f'<div class="section-header">{L["model_config"]}</div>', unsafe_allow_html=True)
    config_col1, config_col2, config_col3, config_col4 = st.columns(4)

    with config_col1:
        st.markdown(f"**{L['data_parameters']}**")
        time_window = st.slider(L["time_window"], 5, 100, 60, help=L["time_window"])
        test_ratio = st.slider(L["test_ratio"], 0.1, 0.5, 0.2, help=L["test_ratio"])

    with config_col2:
        st.markdown(f"**{L['model_architecture']}**")
        model_choice = st.selectbox(L["model_type"], ["LSTM", "MLP", "Hybrid"], help=L["model_type"])
        layers = st.slider(L["hidden_layers"], 1, 5, 3, help=L["hidden_layers"])

    with config_col3:
        st.markdown(f"**{L['training_params']}**")
        epochs = st.slider(L["epochs"], 10, 500, 100, help=L["epochs"])
        batch_size = st.selectbox(L["batch_size"], [16, 32, 64, 128], index=1, help=L["batch_size"])

    with config_col4:
        st.markdown(f"**{L['prediction_settings']}**")
        forecast_days = st.slider(L["forecast_days"], 1, 30, 7, help=L["forecast_days"])
        confidence_level = st.slider(L["confidence_level"], 0.8, 0.99, 0.95, help=L["confidence_level"])

# ------------------------------
# 🏢 TASI Stock Search & Data Management
# ------------------------------
st.markdown(f'<div class="section-header">{L["data_management"]}</div>', unsafe_allow_html=True)

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

data_source = st.radio(
    f"**{L['select_data_source']}**",
    [L["upload_file"], L["tasi_search"]],
    horizontal=True
)

df = None

# Helper: persist df in session_state if loaded
if 'df' not in st.session_state:
    st.session_state.df = None

# ------------------------------
# Load data from TASI
# ------------------------------
if data_source == L["tasi_search"]:
    col_search1, col_search2, col_search3 = st.columns([2, 1, 1])

    with col_search1:
        selected_stock = st.selectbox(
            f"🔍 {L['select_data_source']}", 
            options=list(tasi_stocks.keys()),
            help=L["select_data_source"]
        )

    with col_search2:
        period = st.selectbox(
            L["historical_period"],
            ["3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )

    with col_search3:
        st.markdown("###")
        load_data_btn = st.button(L["load_stock_data"], use_container_width=True)

    if load_data_btn:
        with st.spinner(f"📊 {L['cleaning_info']} {selected_stock}..."):
            try:
                symbol = tasi_stocks[selected_stock]
                stock_data = yf.download(symbol, period=period)

                if stock_data is None or stock_data.empty:
                    st.error(L["no_historical"])
                else:
                    df = stock_data.reset_index()
                    if 'Close' not in df.columns:
                        st.error(L["no_close"])
                    else:
                        st.success(f"✅ Loaded {len(df)} days of historical data for {selected_stock}")
                        st.session_state.df = df.copy()
            except Exception as e:
                st.error(f"{L['error_loading']} {str(e)}")

else:
    # File upload
    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        uploaded_file = st.file_uploader(f"**{L['upload_file']}**", type=["csv", "xlsx"],
                                         help="Upload CSV or Excel file with OHLC data")

    with upload_col2:
        st.markdown(f"**{L['generate_sample_text']}**")
        if st.button(L["sample_data"], use_container_width=True):
            dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
            sample_data = pd.DataFrame({
                'Date': dates,
                'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
                'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
                'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            })
            sample_data['Close'] = sample_data['Close'].abs()
            csv = sample_data.to_csv(index=False)
            st.download_button(L["download_sample"], data=csv, file_name="sample_stock_data.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith("xlsx"):
                xls = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("📑 Select Worksheet", xls.sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()
        except Exception as e:
            st.error(f"{L['error_loading']} {e}")

# Use df either from session_state or local
if st.session_state.df is not None:
    df = st.session_state.df

# ------------------------------
# If data exists, show preview & normalization
# ------------------------------
if df is not None and 'Close' in df.columns:
    st.success(f"✅ Successfully loaded data with {len(df)} records")

    preview_col1, preview_col2 = st.columns([3, 1])
    with preview_col1:
        st.markdown(f"**{L['data_preview']}**")
        st.dataframe(df.head(10), use_container_width=True)

    with preview_col2:
        st.markdown(f"**{L['data_summary']}**")
        st.metric(L["total_records"], len(df))
        if 'Date' in df.columns:
            # Ensure Date is datetime
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                pass
            date_range = f"{df['Date'].min()} to {df['Date'].max()}"
        else:
            date_range = f"Index {df.index.min()} to {df.index.max()}"
        st.metric(L["date_range"], date_range)
        try:
            price_change = float(((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100)
            st.metric(L["total_return"], f"{price_change:.2f}%")
        except:
            st.metric(L["total_return"], "N/A")

    # ------------------------------
    # 🔄 Data normalization (robust/fixed)
    # ------------------------------
    st.markdown(f'<div class="section-header">{L["data_preprocessing"]}</div>', unsafe_allow_html=True)

    def normalize_stock_data(df_in):
        df_local = df_in.copy()
        st.info(L["cleaning_info"])

        # Show available columns
        available_columns = [str(col) for col in df_local.columns.tolist()]
        st.write(f"{L['available_columns']} {', '.join(available_columns)}")

        # Determine columns to normalize
        possible_columns = ["Price", "Open", "High", "Low", "Vol.", "Change %", "Volume"]
        columns_to_normalize = [col for col in possible_columns if col in df_local.columns]
        st.write(f"{L['columns_to_normalize']} {', '.join(columns_to_normalize) if columns_to_normalize else 'None'}")

        # Clean volume column
        volume_column = None
        if "Vol." in df_local.columns:
            volume_column = "Vol."
        elif "Volume" in df_local.columns:
            volume_column = "Volume"

        if volume_column:
            def clean_volume(val):
                try:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return 0.0
                    if isinstance(val, str):
                        val2 = val.replace(",", "").strip()
                        if val2.endswith("B"):
                            return float(val2[:-1]) * 1_000_000_000
                        elif val2.endswith("M"):
                            return float(val2[:-1]) * 1_000_000
                        elif val2.endswith("K"):
                            return float(val2[:-1]) * 1_000
                        else:
                            return float(val2)
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            df_local[volume_column] = [clean_volume(x) for x in df_local[volume_column]]
            st.success(L["cleaned_column"].format(col=volume_column))

        # Clean Change % if present
        if "Change %" in df_local.columns:
            try:
                df_local["Change %"] = df_local["Change %"].astype(str).str.replace("%", "", regex=False)
                df_local["Change %"] = pd.to_numeric(df_local["Change %"], errors='coerce').fillna(0)
                st.success(L["cleaned_column"].format(col="Change %"))
            except Exception as e:
                st.warning(L["cleaning_warning"].format(col="Change %") + str(e))

        # Clean numeric columns
        numeric_columns = ["Price", "Open", "High", "Low"]
        for col in numeric_columns:
            if col in df_local.columns:
                try:
                    df_local[col] = df_local[col].astype(str).str.replace(",", "")
                    df_local[col] = pd.to_numeric(df_local[col], errors='coerce').fillna(0)
                    st.success(L["cleaned_column"].format(col=col))
                except Exception as e:
                    st.warning(L["cleaning_warning"].format(col=col) + str(e))

        # Clean Close column
        if "Close" in df_local.columns:
            try:
                df_local["Close"] = df_local["Close"].astype(str).str.replace(",", "")
                df_local["Close"] = pd.to_numeric(df_local["Close"], errors='coerce')
                df_local = df_local.dropna(subset=['Close'])
                st.success(L["cleaned_column"].format(col="Close"))
            except Exception as e:
                st.error(f"{L['error_loading']} {e}")

        # Apply MinMax scaling to numeric columns (excluding Close for model target)
        if columns_to_normalize:
            scaler = MinMaxScaler()
            valid_columns = [col for col in columns_to_normalize if col in df_local.columns and pd.api.types.is_numeric_dtype(df_local[col])]
            if valid_columns:
                try:
                    df_local[valid_columns] = df_local[valid_columns].fillna(0)
                    df_local[valid_columns] = scaler.fit_transform(df_local[valid_columns])
                    st.success(f"{L['applied_normalization']} {', '.join(valid_columns)}")
                    # Show examples
                    st.markdown(L["normalization_examples"])
                    norm_cols = st.columns(min(3, len(valid_columns)))
                    for i, col in enumerate(valid_columns[:3]):
                        with norm_cols[i]:
                            st.metric(f"{col} (Normalized)", f"{df_local[col].iloc[0]:.3f}")
                except Exception as e:
                    st.error(f"{L['error_loading']} {e}")
            else:
                st.warning(L["no_numeric_cols"])
        else:
            st.warning(L["no_cols_available"])

        return df_local

    # Run normalization
    with st.spinner(L["cleaning_info"]):
        try:
            df_normalized = normalize_stock_data(df)
            df = df_normalized
            st.success(L["normalization_completed"])

            col_norm1, col_norm2 = st.columns([2, 1])
            with col_norm1:
                st.markdown("**📊 Normalized Data Preview**")
                st.dataframe(df.head(10), use_container_width=True)
            with col_norm2:
                st.markdown("**📈 Data Ranges**")
                if 'Close' in df.columns:
                    try:
                        st.metric(L["close_range"], f"{df['Close'].min():.2f} - {df['Close'].max():.2f}")
                    except:
                        st.metric(L["close_range"], "N/A")
                normalized_cols = [col for col in ["Open", "High", "Low", "Vol.", "Volume", "Change %"] if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                for col in normalized_cols[:3]:
                    try:
                        st.metric(f"{col} {L['range']}", f"{df[col].min():.3f} - {df[col].max():.3f}")
                    except:
                        pass
        except Exception as e:
            st.error(f"{L['error_loading']} {e}")
            st.info("⚠️ Continuing with original data...")

    # ------------------------------
    # 🤖 Model Training Section
    # ------------------------------
    st.markdown(f'<div class="section-header">{L["ai_model_training"]}</div>', unsafe_allow_html=True)

    with st.spinner("🚀 Training AI model... This may take a few moments"):
        # Preprocessing - target is Close; scale Close for training
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
                model.add(Dense(max(1, 128 // (i + 1)), activation="relu"))
                model.add(Dropout(0.2))
            model.add(Dense(1))
        elif model_choice == "LSTM":
            model.add(LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.3))
            for i in range(layers - 1):
                model.add(LSTM(max(1, 100 // (i + 1)), return_sequences=(i < layers - 2)))
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

        # Callback to update progress bar - use verbose loop to update periodically (approximate)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )

        progress_bar.progress(100)
        status_text.success(L["training_completed"])

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
    # 🎯 Performance Metrics Display
    # ------------------------------
    st.markdown(f'<div class="section-header">{L["model_performance"]}</div>', unsafe_allow_html=True)

    def get_performance_badge(metric, value):
        if metric == "R²":
            if value >= 0.9: return "excellent"
            elif value >= 0.7: return "good"
            elif value >= 0.5: return "fair"
            else: return "poor"
        elif metric == "RMSE":
            avg_price = np.mean(y_test_inv)
            relative_error = value / avg_price if avg_price != 0 else 1
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
            <div class="metric-title">{L['root_mean_square_error']}</div>
            <div class="metric-value" style="color: #ff6b6b;">{rmse:.2f}</div>
            <span class="performance-badge {rmse_badge}">{rmse_badge.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{L['mean_square_error']}</div>
            <div class="metric-value" style="color: #feca57;">{mse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        r2_badge = get_performance_badge("R²", r2)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{L['r2_score']}</div>
            <div class="metric-value" style="color: #1dd1a1;">{r2:.4f}</div>
            <span class="performance-badge {r2_badge}">{r2_badge.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{L['mean_absolute_percentage_error']}</div>
            <div class="metric-value" style="color: #ff9ff3;">{mape:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------
    # 📈 Visualization Tabs
    # ------------------------------
    st.markdown(f'<div class="section-header">{L["advanced_analytics"]}</div>', unsafe_allow_html=True)
    tabs = st.tabs([L["price_prediction"], L["market_analysis"], L["loss_metrics"], L["future_forecast_tab"]])

    with tabs[0]:
        col_chart1, col_chart2 = st.columns([3, 1])
        with col_chart1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=y_test_inv.flatten(),
                name="Actual Prices",
                line=dict(color="#1dd1a1", width=3),
                mode='lines'
            ))
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
        st.subheader(L["market_analysis_candlestick"])
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # ensure index is dates if we have Date
            try:
                x_index = df['Date'] if 'Date' in df.columns else df.index
            except:
                x_index = df.index
            candlestick = go.Figure(data=[go.Candlestick(
                x=x_index[-100:],
                open=df["Open"].tail(100),
                high=df["High"].tail(100),
                low=df["Low"].tail(100),
                close=df["Close"].tail(100),
                increasing_line_color='#1dd1a1',
                decreasing_line_color='#ff6b6b'
            )])
            candlestick.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500, title="Last 100 Trading Days")
            st.plotly_chart(candlestick, use_container_width=True)
        else:
            st.warning("⚠️ OHLC data required for candlestick chart")

    with tabs[2]:
        col_loss1, col_loss2 = st.columns([3, 1])
        with col_loss1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(history.history["loss"], label="Training Loss", color="#667eea", linewidth=2)
            ax1.plot(history.history["val_loss"], label="Validation Loss", color="#ff6b6b", linewidth=2)
            ax1.set_title("Model Training History", fontsize=14, fontweight='bold', color='white')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#0c0c0c')

            if 'mae' in history.history:
                ax2.plot(history.history["mae"], label="Training MAE", color="#1dd1a1", linewidth=2)
                ax2.plot(history.history["val_mae"], label="Validation MAE", color="#feca57", linewidth=2)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#0c0c0c')

            fig.patch.set_facecolor('#0c0c0c')
            st.pyplot(fig)
        with col_loss2:
            st.markdown(f"**{L['training_summary']}**")
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            st.metric("Final Train Loss", f"{final_loss:.4f}")
            st.metric("Final Val Loss", f"{final_val_loss:.4f}")
            st.metric("Training Epochs", len(history.history['loss']))

    with tabs[3]:
        st.subheader(L["future_price_forecast"])
        last_sequence = scaled[-time_window:]
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(forecast_days):
            next_pred = model.predict(current_sequence.reshape(1, time_window, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]

        future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # create future dates
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        else:
            future_dates = list(range(len(df), len(df) + forecast_days))

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices.flatten(),
            name="Forecast",
            line=dict(color="#667eea", width=3),
            mode='lines+markers'
        ))
        fig_future.update_layout(title=f"{forecast_days}-Day Price Forecast", xaxis_title="Date", yaxis_title="Predicted Price ($)", template="plotly_dark", height=400)
        st.plotly_chart(fig_future, use_container_width=True)

    # ------------------------------
    # 💾 Export Results
    # ------------------------------
    st.markdown(f'<div class="section-header">{L["export_results"]}</div>', unsafe_allow_html=True)

    pred_df = pd.DataFrame({
        "Actual_Price": y_test_inv.flatten(),
        "Predicted_Price": y_pred_inv.flatten(),
        "Absolute_Error": np.abs(y_test_inv.flatten() - y_pred_inv.flatten()),
        "Percentage_Error": (np.abs(y_test_inv.flatten() - y_pred_inv.flatten()) / y_test_inv.flatten()) * 100
    })

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        csv = pred_df.to_csv(index=False)
        st.download_button(L["download_predictions"], data=csv, file_name="stock_predictions.csv", mime="text/csv", use_container_width=True)

    with col_export2:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary_text = "\n".join(model_summary)
        st.download_button(L["model_architecture_file"], data=model_summary_text, file_name="model_architecture.txt", mime="text/plain", use_container_width=True)

    with col_export3:
        report = f"""
        Stock Prediction
