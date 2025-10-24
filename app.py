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

# Language selection
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Optional page config
st.set_page_config(page_title="Stock Prediction Center", layout="wide")

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

    /* Professional header (kept but we will render a smaller inline header instead of this large block) */
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
    
    /* Custom header with animated gradient - kept for potential use but we will render a smaller title below */
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

    /* make the inline header text smaller when we render it as gradient text */
    .inline-gradient-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    
    /* Learn more section styling */
    .learn-more-content {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        border-left: 4px solid #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Language texts
texts = {
    'English': {
        'title': '📈 Stock Prediction Center',
        'status': 'Status',
        'ready': 'Ready',
        'online': 'Online',
        'last_update': 'Last Update',
        'config_title': '⚙️ Model Configuration',
        'data_params': '📊 Data Parameters',
        'time_window': 'Time Window (days)',
        'time_window_help': 'Number of days to look back for prediction',
        'test_ratio': 'Test Ratio',
        'test_ratio_help': 'Proportion of data for testing',
        'model_arch': '🤖 Model Architecture',
        'model_type': 'Model Type',
        'model_help': 'Choose neural network architecture',
        'layers': 'Hidden Layers',
        'layers_help': 'Number of hidden layers',
        'training_params': '🔧 Training Parameters',
        'epochs': 'Training Epochs',
        'epochs_help': 'Number of training iterations',
        'batch_size': 'Batch Size',
        'batch_help': 'Training batch size',
        'prediction_settings': '📈 Prediction Settings',
        'forecast_days': 'Forecast Days',
        'forecast_help': 'Days to forecast into future',
        'confidence': 'Confidence Level',
        'confidence_help': 'Prediction confidence interval',
        'data_management': '📁 Data Management',
        'data_source': 'Select Data Source:',
        'upload_file': '📤 Upload File',
        'tasi_search': '🏢 TASI Stock Search',
        'select_tasi': '🔍 Select TASI Company',
        'tasi_help': 'Choose a company from TASI to load historical stock data',
        'historical_period': '📅 Historical Period',
        'load_data': '📥 Load Stock Data',
        'upload_market': 'Upload Market Data',
        'upload_help': 'Upload CSV or Excel file with OHLC data',
        'sample_data': 'Sample Data',
        'generate_sample': '🎲 Generate Sample Data',
        'download_sample': '📥 Download Sample',
        'data_preview': '🔍 Data Preview',
        'data_summary': '📋 Data Summary',
        'total_records': 'Total Records',
        'date_range': 'Date Range',
        'total_return': 'Total Return',
        'preprocessing': '🔄 Data Preprocessing',
        'model_training': '🤖 AI Model Training',
        'performance': '📊 Model Performance',
        'analytics': '📈 Advanced Analytics',
        'price_prediction': '📊 Price Prediction',
        'market_analysis': '🕯️ Market Analysis',
        'loss_metrics': '📉 Loss Metrics',
        'future_forecast': '🔮 Future Forecast',
        'export': '💾 Export Results',
        'download_predictions': '📥 Download Predictions',
        'model_arch_download': '📋 Model Architecture',
        'training_report': '📄 Training Report',
        'welcome_title': '🚀 Welcome to AI Stock Predictor',
        'welcome_text': 'Choose a data source above to get started with AI-powered stock predictions. Load TASI stock data automatically or upload your own historical market data.',
        'how_it_works': '📊 How it works:',
        'select_stock': 'Select Stock',
        'select_stock_desc': 'Choose from TASI companies or upload your data',
        'ai_training': 'AI Training',
        'ai_training_desc': 'Neural networks learn market patterns',
        'get_predictions': 'Get Predictions',
        'get_predictions_desc': 'Receive accurate price forecasts',
        'footer': '© 2024 Stock Prediction Center | Built with Streamlit & TensorFlow',
        'learn_more': '📚 Learn More',
        'learn_more_title': '📚 How This Stock Prediction System Works',
        'learn_more_content': '''
        <div class="learn-more-content">
            <h3>🔍 How the System Works</h3>
            
            <h4>📊 Data Processing</h4>
            <p>The system processes historical stock data through several steps:</p>
            <ul>
                <li><strong>Data Collection:</strong> Fetches real-time TASI stock data or accepts uploaded files</li>
                <li><strong>Data Cleaning:</strong> Handles missing values, converts formats, and normalizes data</li>
                <li><strong>Normalization:</strong> Scales all values between 0-1 using MinMaxScaler for better model performance</li>
            </ul>
            
            <h4>🤖 AI Model Architecture</h4>
            <p>Three different neural network models are available:</p>
            <ul>
                <li><strong>LSTM:</strong> Long Short-Term Memory networks ideal for time series data</li>
                <li><strong>MLP:</strong> Multi-Layer Perceptron for simpler patterns</li>
                <li><strong>Hybrid:</strong> Combines LSTM and dense layers for complex patterns</li>
            </ul>
            
            <h4>🎯 Training Process</h4>
            <ul>
                <li>Data is split into training (80%) and testing (20%) sets</li>
                <li>Model learns from historical price patterns</li>
                <li>Early stopping prevents overfitting</li>
                <li>Multiple epochs refine prediction accuracy</li>
            </ul>
            
            <h4>📈 Prediction & Analysis</h4>
            <ul>
                <li>Generates future price forecasts</li>
                <li>Provides confidence intervals</li>
                <li>Visualizes predictions vs actual prices</li>
                <li>Calculates performance metrics (RMSE, R², MAPE)</li>
            </ul>
            
            <h4>💡 How to Use</h4>
            <ol>
                <li>Select your data source (TASI stocks or file upload)</li>
                <li>Configure model parameters in the settings</li>
                <li>Load and preview your data</li>
                <li>Train the AI model</li>
                <li>Analyze predictions and export results</li>
            </ol>
            
            <p><strong>Note:</strong> Stock predictions are based on historical patterns and should be used as one of many tools in your investment decision process.</p>
        </div>
        '''
    },
    'Arabic': {
        'title': '📈 مركز توقع الأسهم',
        'status': 'الحالة',
        'ready': 'جاهز',
        'online': 'متصل',
        'last_update': 'آخر تحديث',
        'config_title': '⚙️ إعدادات النموذج',
        'data_params': '📊 معاملات البيانات',
        'time_window': 'نافذة الوقت (أيام)',
        'time_window_help': 'عدد الأيام للنظر إليها للتنبؤ',
        'test_ratio': 'نسبة الاختبار',
        'test_ratio_help': 'نسبة البيانات المستخدمة للاختبار',
        'model_arch': '🤖 بنية النموذج',
        'model_type': 'نوع النموذج',
        'model_help': 'اختر بنية الشبكة العصبية',
        'layers': 'الطبقات المخفية',
        'layers_help': 'عدد الطبقات المخفية',
        'training_params': '🔧 معاملات التدريب',
        'epochs': 'دورات التدريب',
        'epochs_help': 'عدد دورات التدريب',
        'batch_size': 'حجم الدفعة',
        'batch_help': 'حجم الدفعة للتدريب',
        'prediction_settings': '📈 إعدادات التنبؤ',
        'forecast_days': 'أيام التنبؤ',
        'forecast_help': 'عدد الأيام للتنبؤ بالمستقبل',
        'confidence': 'مستوى الثقة',
        'confidence_help': 'فترة ثقة التنبؤ',
        'data_management': '📁 إدارة البيانات',
        'data_source': 'اختر مصدر البيانات:',
        'upload_file': '📤 رفع ملف',
        'tasi_search': '🏢 بحث أسهم تاسي',
        'select_tasi': '🔍 اختر شركة تاسي',
        'tasi_help': 'اختر شركة من تاسي لتحميل البيانات التاريخية',
        'historical_period': '📅 الفترة التاريخية',
        'load_data': '📥 تحميل بيانات الأسهم',
        'upload_market': 'رفع بيانات السوق',
        'upload_help': 'رفع ملف CSV أو Excel يحتوي على بيانات OHLC',
        'sample_data': 'بيانات نموذجية',
        'generate_sample': '🎲 إنشاء بيانات نموذجية',
        'download_sample': '📥 تحميل النموذج',
        'data_preview': '🔍 معاينة البيانات',
        'data_summary': '📋 ملخص البيانات',
        'total_records': 'إجمالي السجلات',
        'date_range': 'النطاق الزمني',
        'total_return': 'إجمالي العائد',
        'preprocessing': '🔄 معالجة البيانات',
        'model_training': '🤖 تدريب النموذج الذكي',
        'performance': '📊 أداء النموذج',
        'analytics': '📈 التحليلات المتقدمة',
        'price_prediction': '📊 تنبؤ الأسعار',
        'market_analysis': '🕯️ تحليل السوق',
        'loss_metrics': '📉 مقاييس الخسارة',
        'future_forecast': '🔮 التنبؤ المستقبلي',
        'export': '💾 تصدير النتائج',
        'download_predictions': '📥 تحميل التنبؤات',
        'model_arch_download': '📋 بنية النموذج',
        'training_report': '📄 تقرير التدريب',
        'welcome_title': '🚀 مرحباً بكم في منصة توقع الأسهم الذكية',
        'welcome_text': 'اختر مصدر البيانات أعلاه للبدء في التنبؤ بالأسهم باستخدام الذكاء الاصطناعي. قم بتحميل بيانات أسهم تاسي تلقائياً أو ارفع بيانات السوق التاريخية الخاصة بك.',
        'how_it_works': '📊 كيف يعمل:',
        'select_stock': 'اختيار السهم',
        'select_stock_desc': 'اختر من شركات تاسي أو ارفع بياناتك',
        'ai_training': 'التدريب الذكي',
        'ai_training_desc': 'الشبكات العصبية تتعلم أنماط السوق',
        'get_predictions': 'الحصول على تنبؤات',
        'get_predictions_desc': 'احصل على تنبؤات دقيقة للأسعار',
        'footer': '© 2024 مركز توقع الأسهم | مبنى باستخدام Streamlit & TensorFlow',
        'learn_more': '📚 اعرف المزيد',
        'learn_more_title': '📚 كيف يعمل نظام توقع الأسهم هذا',
        'learn_more_content': '''
        <div class="learn-more-content" style="text-align: right; direction: rtl;">
            <h3>🔍 كيف يعمل النظام</h3>
            
            <h4>📊 معالجة البيانات</h4>
            <p>يقوم النظام بمعالجة بيانات الأسهم التاريخية من خلال عدة خطوات:</p>
            <ul>
                <li><strong>جمع البيانات:</strong> يجلب بيانات أسهم تاسي في الوقت الحقيقي أو يقبل الملفات المرفوعة</li>
                <li><strong>تنظيف البيانات:</strong> يتعامل مع القيم المفقودة، يحول التنسيقات، ويطبع البيانات</li>
                <li><strong>التطبيع:</strong> يقيس جميع القيم بين 0-1 باستخدام MinMaxScaler لأداء أفضل للنموذج</li>
            </ul>
            
            <h4>🤖 بنية النموذج الذكي</h4>
            <p>ثلاثة نماذج مختلفة للشبكات العصبية متاحة:</p>
            <ul>
                <li><strong>LSTM:</strong> شبكات الذاكرة طويلة المدى المثالية لبيانات السلاسل الزمنية</li>
                <li><strong>MLP:</strong>多层感知器 للأنماط البسيطة</li>
                <li><strong>Hybrid:</strong> يجمع بين طبقات LSTM والطبقات الكثيفة للأنماط المعقدة</li>
            </ul>
            
            <h4>🎯 عملية التدريب</h4>
            <ul>
                <li>يتم تقسيم البيانات إلى مجموعات تدريب (80٪) واختبار (20٪)</li>
                <li>يتعلم النموذج من أنماط الأسعار التاريخية</li>
                <li>التوقف المبكر يمنع الإفراط في التجهيز</li>
                <li>دورات متعددة تحسن دقة التنبؤ</li>
            </ul>
            
            <h4>📈 التنبؤ والتحليل</h4>
            <ul>
                <li>يولد تنبؤات الأسعار المستقبلية</li>
                <li>يوفر فترات ثقة</li>
                <li>يصور التنبؤات مقابل الأسعار الفعلية</li>
                <li>يحسب مقاييس الأداء (RMSE, R², MAPE)</li>
            </ul>
            
            <h4>💡 كيفية الاستخدام</h4>
            <ol>
                <li>اختر مصدر البيانات (أسهم تاسي أو رفع ملف)</li>
                <li>اضبط معاملات النموذج في الإعدادات</li>
                <li>قم بتحميل ومعاينة بياناتك</li>
                <li>درب النموذج الذكي</li>
                <li>حلل التنبؤات وقم بتصدير النتائج</li>
            </ol>
            
            <p><strong>ملاحظة:</strong> تنبؤات الأسهم تستند إلى الأنماط التاريخية ويجب استخدامها كأحد الأدوات العديدة في عملية قرار الاستثمار.</p>
        </div>
        '''
    }
}

# Language selector
col_lang, col_header1, col_header2, col_header3 = st.columns([1, 3, 1, 1])
with col_lang:
    lang = st.selectbox("🌐 Language / اللغة", ["English", "Arabic"], key="lang_selector")
    st.session_state.language = lang

current_lang = st.session_state.language
t = texts[current_lang]

with col_header1:
    st.markdown(f'<div class="inline-gradient-title">{t["title"]}</div>', unsafe_allow_html=True)

with col_header2:
    st.metric(t["status"], t["ready"], delta=t["online"])

with col_header3:
    st.metric(t["last_update"], datetime.now().strftime("%H:%M"))

# Learn More Section
with st.expander(t["learn_more"], expanded=False):
    st.markdown(t["learn_more_content"], unsafe_allow_html=True)

# ------------------------------
# 🎯 Configuration Panel (moved into collapsible expander)
# ------------------------------
# Use an expander collapsed by default so user can click arrow to reveal settings
with st.expander(t["config_title"], expanded=False):
    # optional section header inside expander to keep visual style
    st.markdown(f'<div class="section-header">{t["config_title"]}</div>', unsafe_allow_html=True)

    config_col1, config_col2, config_col3, config_col4 = st.columns(4)

    with config_col1:
        st.markdown(f"**{t['data_params']}**")
        time_window = st.slider(t["time_window"], 5, 100, 60, help=t["time_window_help"])
        test_ratio = st.slider(t["test_ratio"], 0.1, 0.5, 0.2, help=t["test_ratio_help"])

    with config_col2:
        st.markdown(f"**{t['model_arch']}**")
        model_choice = st.selectbox(t["model_type"], ["LSTM", "MLP", "Hybrid"], help=t["model_help"])
        layers = st.slider(t["layers"], 1, 5, 3, help=t["layers_help"])

    with config_col3:
        st.markdown(f"**{t['training_params']}**")
        epochs = st.slider(t["epochs"], 10, 500, 100, help=t["epochs_help"])
        batch_size = st.selectbox(t["batch_size"], [16, 32, 64, 128], index=1, help=t["batch_help"])

    with config_col4:
        st.markdown(f"**{t['prediction_settings']}**")
        forecast_days = st.slider(t["forecast_days"], 1, 30, 7, help=t["forecast_help"])
        confidence_level = st.slider(t["confidence"], 0.8, 0.99, 0.95, help=t["confidence_help"])

# ------------------------------
# 🏢 TASI Stock Search & Data Management
# ------------------------------
st.markdown(f'<div class="section-header">{t["data_management"]}</div>', unsafe_allow_html=True)

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
    f"**{t['data_source']}**",
    [t["upload_file"], t["tasi_search"]],
    horizontal=True
)

df = None

if data_source == t["tasi_search"]:
    col_search1, col_search2, col_search3 = st.columns([2, 1, 1])
    
    with col_search1:
        selected_stock = st.selectbox(
            t["select_tasi"], 
            options=list(tasi_stocks.keys()),
            help=t["tasi_help"]
        )
    
    with col_search2:
        period = st.selectbox(
            t["historical_period"],
            ["3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
    
    with col_search3:
        st.markdown("###")  # Vertical spacing
        load_data = st.button(t["load_data"], use_container_width=True)
    
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
        uploaded_file = st.file_uploader(f"**{t['upload_market']}**", type=["csv", "xlsx"], 
                                       help=t["upload_help"])

    with upload_col2:
        st.markdown(f"**{t['sample_data']}**")
        if st.button(t["generate_sample"], use_container_width=True):
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
            st.download_button(t["download_sample"], data=csv, file_name="sample_stock_data.csv", mime="text/csv")

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
        st.markdown(f"**{t['data_preview']}**")
        st.dataframe(df.head(10), use_container_width=True)
    
    with preview_col2:
        st.markdown(f"**{t['data_summary']}**")
        st.metric(t["total_records"], len(df))
        if 'Date' in df.columns:
            date_range = f"{df['Date'].min()} to {df['Date'].max()}"
        else:
            date_range = f"Index {df.index.min()} to {df.index.max()}"
        st.metric(t["date_range"], date_range)
        try:
            price_change = float(((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100)
            st.metric(t["total_return"], f"{price_change:.2f}%")
        except:
            st.metric(t["total_return"], "N/A")

    # ------------------------------
    # 🔄 DATA NORMALIZATION - COMPLETELY FIXED VERSION
    # ------------------------------
    st.markdown(f'<div class="section-header">{t["preprocessing"]}</div>', unsafe_allow_html=True)
    
    def normalize_stock_data(df):
        """
        Simple and safe normalization for yfinance data
        """
        df = df.copy()
        
        st.info("🔄 Cleaning stock data...")
        
        try:
            # Clean the Close column (most important for predictions)
            if 'Close' in df.columns:
                # Convert to numeric and handle errors
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                # Remove rows where Close is NaN after conversion
                df = df.dropna(subset=['Close'])
                st.success("✅ Cleaned Close column")
            
            # Just ensure other numeric columns are clean but don't normalize them
            # For yfinance data, we don't need complex normalization
            numeric_columns = ['Open', 'High', 'Low', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            st.success("✅ Data cleaning completed successfully!")
            
            # Show data info - FIXED: Use proper string formatting
            st.write(f"📊 Cleaned data shape: {df.shape}")
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            st.write(f"🔢 Available numeric columns: {numeric_cols}")
            
        except Exception as e:
            st.error(f"❌ Error during data cleaning: {str(e)}")
            st.info("⚠️ Continuing with original data...")
        
        return df

    # Apply normalization
    with st.spinner("🔄 Processing stock data..."):
        try:
            df_cleaned = normalize_stock_data(df)
            df = df_cleaned
            st.success("✅ Data processing completed!")
            
            # Show cleaned data preview
            col_norm1, col_norm2 = st.columns([2, 1])
            
            with col_norm1:
                st.markdown("**📊 Processed Data Preview**")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col_norm2:
                st.markdown("**📈 Data Statistics**")
                # Show Close price range - FIXED: Use proper string formatting
                if 'Close' in df.columns:
                    close_min = float(df['Close'].min())
                    close_max = float(df['Close'].max())
                    st.metric(
                        "Close Price Range", 
                        f"${close_min:.2f} - ${close_max:.2f}"
                    )
                
                # Show other columns info
                if 'Volume' in df.columns:
                    volume_mean = float(df['Volume'].mean())
                    st.metric("Avg Volume", f"{volume_mean:.0f}")
                
        except Exception as e:
            st.error(f"❌ Data processing failed: {str(e)}")
            st.info("⚠️ Continuing with original data...")

    # ------------------------------
    # 🤖 Model Training Section
    # ------------------------------
    st.markdown(f'<div class="section-header">{t["model_training"]}</div>', unsafe_allow_html=True)
    
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
    st.markdown(f'<div class="section-header">{t["performance"]}</div>', unsafe_allow_html=True)
    
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
    st.markdown(f'<div class="section-header">{t["analytics"]}</div>', unsafe_allow_html=True)
    
    tabs = st.tabs([t["price_prediction"], t["market_analysis"], t["loss_metrics"], t["future_forecast"]])

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
    st.markdown(f'<div class="section-header">{t["export"]}</div>', unsafe_allow_html=True)
    
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
            t["download_predictions"], 
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
            t["model_arch_download"], 
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
        
        Dataset: {uploaded_file.name if data_source == t['upload_file'] else selected_stock}
        Records: {len(df)}
        """
        st.download_button(
            t["training_report"], 
            data=report, 
            file_name="training_report.txt", 
            mime="text/plain",
            use_container_width=True
        )

else:
    # Welcome state with sample visualization
    st.markdown(f"""
    <div style='text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.05); border-radius: 20px; margin: 2rem 0;'>
        <h2 style='color: #667eea; margin-bottom: 1rem;'>{t['welcome_title']}</h2>
        <p style='color: #b0b0b0; font-size: 1.2rem; max-width: 600px; margin: 0 auto;'>
            {t['welcome_text']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample visualization
    st.markdown(f"### {t['how_it_works']}")
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🏢</div>
            <h4>{t['select_stock']}</h4>
            <p style='color: #b0b0b0;'>{t['select_stock_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_demo2:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
            <h4>{t['ai_training']}</h4>
            <p style='color: #b0b0b0;'>{t['ai_training_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_demo3:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🔮</div>
            <h4>{t['get_predictions']}</h4>
            <p style='color: #b0b0b0;'>{t['get_predictions_desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# 📱 Footer
# ------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.9rem;'>{t['footer']}</div>", 
        unsafe_allow_html=True
    )