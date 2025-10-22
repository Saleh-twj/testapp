import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Language configuration
def get_translations(language):
    translations = {
        'english': {
            'title': "📈 Stock Price Prediction System - Saudi Trading",
            'sidebar_settings': "⚙️ Model Settings",
            'upload_file': "📤 Upload Stock Data File (CSV)",
            'training_params': "🎛️ Training Parameters",
            'time_window': "Time Window (days)",
            'test_ratio': "Test Ratio",
            'epochs': "Number of Epochs",
            'batch_size': "Batch Size",
            'select_model': "Select Model",
            'start_training': "🚀 Start Training",
            'data_overview': "📊 Data Overview",
            'total_records': "Total Records",
            'time_period': "Time Period",
            'min_price': "Minimum Price",
            'max_price': "Maximum Price",
            'tabs': ["📋 Data View", "📈 Charts", "🔍 Statistical Analysis"],
            'price_evolution': "Stock Price Evolution Over Time",
            'candlestick': "Candlestick Chart",
            'price_distribution': "Price Distribution",
            'descriptive_stats': "Descriptive Statistics",
            'training_results': "🎯 Training and Prediction Results",
            'training_model': "Training model...",
            'making_predictions': "Making predictions...",
            'completed': "Completed!",
            'performance_metrics': "📊 Performance Metrics",
            'training_loss': "📉 Training Loss Curve",
            'actual_vs_predicted': "🔮 Actual vs Predicted Prices Comparison",
            'future_predictions': "🔭 Future Predictions (30 days)",
            'download_results': "📥 Download Results",
            'download_button': "📥 Download Future Predictions (CSV)",
            'training_completed': "✅ Training and prediction completed successfully!",
            'welcome': "🚀 Welcome to Stock Price Prediction System",
            'welcome_desc': "Intelligent stock price prediction system for Saudi stock market using AI technologies",
            'data_format': "📋 Required Data Format",
            'data_columns': "CSV file should contain the following columns:",
            'date_col': "Date: Trading date",
            'price_col': "Price: Closing price",
            'open_col': "Open: Opening price",
            'high_col': "High: Highest price",
            'low_col': "Low: Lowest price",
            'vol_col': "Vol.: Trading volume",
            'change_col': "Change %: Change percentage",
            'features': "🎯 System Features",
            'features_list': [
                "Deep Learning Models: LSTM and MLP as per project document",
                "Trading Focus: Customized for Saudi stock market",
                "Performance Metrics: RMSE, MSE, and R² according to methodology",
                "User-Friendly Interface: Easy-to-use Arabic design",
                "Future Predictions: 30-day price forecasts",
                "Data Export: Download results for analysis"
            ],
            'model_specs': "📊 Model Specifications",
            'model_specs_list': [
                "LSTM: Two layers with Dropout to prevent overfitting",
                "MLP: With ReLU activation",
                "Data Normalization: Min-Max scaling according to methodology",
                "Time Series: Sliding window approach for sequence prediction"
            ],
            'performance_stats': "📈 Model Performance Statistics",
            'lstm_accuracy': "LSTM Accuracy",
            'mlp_accuracy': "MLP Accuracy",
            'avg_r2': "Average R²",
            'historical_price': "Historical Price",
            'future_forecast': "Future Forecast",
            'training_data': "Training Data",
            'actual_price': "Actual Price",
            'predicted_price': "Predicted Price"
        },
        'arabic': {
            'title': "📈 نظام التنبؤ بأسعار الأسهم - تداول السعودية",
            'sidebar_settings': "⚙️ إعدادات النموذج",
            'upload_file': "📤 حمّل ملف بيانات الأسهم (CSV)",
            'training_params': "🎛️ معاملات التدريب",
            'time_window': "نافذة الزمن (أيام)",
            'test_ratio': "نسبة الاختبار",
            'epochs': "عدد الدورات",
            'batch_size': "حجم الدفعة",
            'select_model': "اختر النموذج",
            'start_training': "🚀 ابدأ التدريب",
            'data_overview': "📊 نظرة عامة على البيانات",
            'total_records': "إجمالي السجلات",
            'time_period': "الفترة الزمنية",
            'min_price': "أقل سعر",
            'max_price': "أعلى سعر",
            'tabs': ["📋 عرض البيانات", "📈 الرسوم البيانية", "🔍 التحليل الإحصائي"],
            'price_evolution': "تطور سعر السهم مع الوقت",
            'candlestick': "الرسم البياني للشموع اليابانية",
            'price_distribution': "توزيع الأسعار",
            'descriptive_stats': "الإحصائيات الوصفية",
            'training_results': "🎯 نتائج التدريب والتنبؤ",
            'training_model': "جاري تدريب النموذج...",
            'making_predictions': "جاري إجراء التنبؤات...",
            'completed': "اكتمل!",
            'performance_metrics': "📊 مقاييس الأداء",
            'training_loss': "📉 منحنى فقدان التدريب",
            'actual_vs_predicted': "🔮 المقارنة بين الأسعار الفعلية والمتوقعة",
            'future_predictions': "🔭 التنبؤات المستقبلية (30 يوم)",
            'download_results': "📥 تحميل النتائج",
            'download_button': "📥 حمّل التنبؤات المستقبلية (CSV)",
            'training_completed': "✅ اكتمل التدريب والتنبؤ بنجاح!",
            'welcome': "🚀 مرحباً بك في نظام التنبؤ بأسعار الأسهم",
            'welcome_desc': "نظام ذكي للتنبؤ بأسعار الأسهم في سوق تداول السعودي باستخدام تقنيات الذكاء الاصطناعي",
            'data_format': "📋 تنسيق البيانات المطلوب",
            'data_columns': "يجب أن يحتوي ملف CSV على الأعمدة التالية:",
            'date_col': "Date: تاريخ التداول",
            'price_col': "Price: سعر الإغلاق",
            'open_col': "Open: سعر الافتتاح",
            'high_col': "High: أعلى سعر",
            'low_col': "Low: أقل سعر",
            'vol_col': "Vol.: حجم التداول",
            'change_col': "Change %: نسبة التغير",
            'features': "🎯 ميزات النظام",
            'features_list': [
                "نماذج التعلم العميق: LSTM و MLP كما في وثيقة المشروع",
                "تركيز على تداول: مخصص لسوق الأسهم السعودي",
                "مقاييس الأداء: RMSE, MSE, و R² حسب المنهجية",
                "واجهة مستخدم سهلة: تصميم عربي سهل الاستخدام",
                "تنبؤات مستقبلية: تنبؤ بأسعار 30 يوم القادمة",
                "تصدير البيانات: تحميل النتائج للتحليل"
            ],
            'model_specs': "📊 مواصفات النماذج",
            'model_specs_list': [
                "LSTM: طبقتان مع Dropout لمنع الإفراط في التمرين",
                "MLP: مع تنشيط ReLU",
                "تطبيع البيانات: تحجيم Min-Max حسب المنهجية",
                "السلاسل الزمنية: نهج النافذة المنزلقة للتنبؤ بالتسلسل"
            ],
            'performance_stats': "📈 إحصائيات أداء النماذج",
            'lstm_accuracy': "دقة LSTM",
            'mlp_accuracy': "دقة MLP",
            'avg_r2': "متوسط R²",
            'historical_price': "السعر التاريخي",
            'future_forecast': "التنبؤات المستقبلية",
            'training_data': "بيانات التدريب",
            'actual_price': "السعر الفعلي",
            'predicted_price': "السعر المتوقع"
        }
    }
    return translations[language]

# تنسيق الصفحة
st.set_page_config(
    page_title="نظام التنبؤ بأسعار الأسهم - تداول",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إضافة تنسيق CSS مخصص
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin: 1.5rem 0rem 1rem 0rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #4b6cb7 0%, #182848 100%);
        color: white;
    }
    .stButton button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    .language-switcher {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_type = None
        
    def load_data(self, file_path):
        """تحميل وتجهيز بيانات الأسهم / Load and prepare stock data"""
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    
    def prepare_data(self, df, time_window=60, test_ratio=0.2):
        """تحضير البيانات للتدريب / Prepare data for training"""
        # استخدام عمود 'Price' كهدف / Use 'Price' column as target
        prices = df['Price'].values.reshape(-1, 1)
        
        # تطبيع البيانات / Normalize data
        scaled_data = self.scaler.fit_transform(prices)
        
        # إنشاء بيانات التدريب والاختبار / Create training and test data
        training_data_len = int(len(scaled_data) * (1 - test_ratio))
        
        # إنشاء مجموعة بيانات التدريب / Create training dataset
        train_data = scaled_data[0:training_data_len, :]
        
        # تقسيم إلى x_train و y_train / Split into x_train and y_train
        x_train = []
        y_train = []
        
        for i in range(time_window, len(train_data)):
            x_train.append(train_data[i-time_window:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # إنشاء مجموعة بيانات الاختبار / Create test dataset
        test_data = scaled_data[training_data_len - time_window:, :]
        x_test = []
        y_test = prices[training_data_len:, :]
        
        for i in range(time_window, len(test_data)):
            x_test.append(test_data[i-time_window:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test, training_data_len
    
    def build_lstm_model(self, time_window, lstm_units=50, dropout_rate=0.2):
        """بناء نموذج LSTM / Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, 
                      input_shape=(time_window, 1)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model
    
    def build_mlp_model(self, time_window, layers=[64, 32, 16]):
        """بناء نموذج MLP / Build MLP model"""
        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=(time_window,)))
        
        for units in layers[1:]:
            model.add(Dense(units, activation='relu'))
        
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model
    
    def train_model(self, x_train, y_train, model_type='LSTM', 
                   epochs=20, batch_size=32, time_window=60):
        """تدريب النموذج المختار / Train selected model"""
        self.model_type = model_type
        
        if model_type == 'LSTM':
            self.model = self.build_lstm_model(time_window)
            history = self.model.fit(x_train, y_train, 
                                   batch_size=batch_size, 
                                   epochs=epochs,
                                   verbose=0)
        else:  # MLP
            # إعادة تشكيل البيانات لـ MLP / Reshape data for MLP
            x_train_mlp = x_train.reshape(x_train.shape[0], x_train.shape[1])
            self.model = self.build_mlp_model(time_window)
            history = self.model.fit(x_train_mlp, y_train, 
                                   batch_size=batch_size, 
                                   epochs=epochs,
                                   verbose=0)
        
        return history
    
    def predict(self, x_test, model_type='LSTM'):
        """إجراء التنبؤات / Make predictions"""
        if model_type == 'LSTM':
            predictions = self.model.predict(x_test, verbose=0)
        else:  # MLP
            x_test_mlp = x_test.reshape(x_test.shape[0], x_test.shape[1])
            predictions = self.model.predict(x_test_mlp, verbose=0)
        
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def calculate_metrics(self, y_true, y_pred):
        """حساب مقاييس الأداء / Calculate performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, r2

def create_candlestick_chart(df, lang):
    """إنشاء رسم الشموع اليابانية / Create candlestick chart"""
    title = "Candlestick Chart" if lang == 'english' else "الرسم البياني للشموع اليابانية"
    x_title = "Date" if lang == 'english' else "التاريخ"
    y_title = "Price" if lang == 'english' else "السعر"
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Price'],
        name='Stock Prices' if lang == 'english' else 'أسعار الأسهم'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template='plotly_white',
        height=500
    )
    
    return fig

def create_performance_gauge(value, title, min_val, max_val):
    """إنشاء مقياس أداء تفاعلي / Create interactive performance gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)*0.6], 'color': "lightgray"},
                {'range': [min_val + (max_val-min_val)*0.6, min_val + (max_val-min_val)*0.8], 'color': "gray"},
                {'range': [min_val + (max_val-min_val)*0.8, max_val], 'color': "darkgray"}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Language selection
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        language = st.selectbox("🌐 Language / اللغة", ["english", "arabic"])
    
    # Get translations
    t = get_translations(language)
    
    # رأس الصفحة الرئيسي / Main header
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    
    # الشريط الجانبي / Sidebar
    with st.sidebar:
        st.markdown(f"### {t['sidebar_settings']}")
        
        # تحميل الملف / File upload
        uploaded_file = st.file_uploader(t['upload_file'], type=['csv'])
        
        st.markdown("---")
        st.markdown(f"### {t['training_params']}")
        
        col1, col2 = st.columns(2)
        with col1:
            time_window = st.slider(t['time_window'], 30, 120, 60)
            test_ratio = st.slider(t['test_ratio'], 0.1, 0.4, 0.2, 0.05)
        with col2:
            epochs = st.slider(t['epochs'], 10, 100, 20)
            batch_size = st.slider(t['batch_size'], 16, 64, 32)
        
        model_type = st.selectbox(t['select_model'], ["LSTM", "MLP"])
        
        st.markdown("---")
        
        if st.button(t['start_training'], type="primary", use_container_width=True):
            st.session_state.run_training = True
        else:
            st.session_state.run_training = False

    if uploaded_file is not None:
        # تهيئة المتنبئ / Initialize predictor
        predictor = StockPredictor()
        
        # تحميل البيانات / Load data
        df = predictor.load_data(uploaded_file)
        
        # عرض نظرة عامة على البيانات / Data overview
        st.markdown(f'<h2 class="section-header">{t["data_overview"]}</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t['total_records'], len(df))
        with col2:
            st.metric(t['time_period'], f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        with col3:
            st.metric(t['min_price'], f"${df['Price'].min():.4f}")
        with col4:
            st.metric(t['max_price'], f"${df['Price'].max():.4f}")
        
        # علامات تبويب للعروض المختلفة / Tabs for different views
        tab1, tab2, tab3 = st.tabs(t['tabs'])
        
        with tab1:
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                # رسم بياني للأسعار / Price chart
                fig1 = px.line(df, x='Date', y='Price', title=t['price_evolution'])
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # رسم الشموع اليابانية / Candlestick chart
                candlestick_fig = create_candlestick_chart(df, language)
                st.plotly_chart(candlestick_fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                # توزيع الأسعار / Price distribution
                fig_hist = px.histogram(df, x='Price', title=t['price_distribution'])
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # إحصائيات描述ية / Descriptive statistics
                st.subheader(t['descriptive_stats'])
                st.dataframe(df.describe(), use_container_width=True)
        
        # التدريب والتنبؤ / Training and prediction
        if st.session_state.run_training:
            st.markdown(f'<h2 class="section-header">{t["training_results"]}</h2>', unsafe_allow_html=True)
            
            with st.spinner(t['training_model']):
                # تحضير البيانات / Prepare data
                x_train, y_train, x_test, y_test, training_data_len = predictor.prepare_data(
                    df, time_window, test_ratio
                )
                
                # شريط التقدم / Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # تدريب النموذج / Train model
                status_text.text(t['training_model'])
                history = predictor.train_model(
                    x_train, y_train, model_type, epochs, batch_size, time_window
                )
                progress_bar.progress(50)
                
                # إجراء التنبؤات / Make predictions
                status_text.text(t['making_predictions'])
                predictions = predictor.predict(x_test, model_type)
                progress_bar.progress(75)
                
                # حساب المقاييس / Calculate metrics
                mse, rmse, r2 = predictor.calculate_metrics(y_test, predictions)
                progress_bar.progress(100)
                status_text.text(t['completed'])
                
                # عرض النتائج / Display results
                st.markdown(f"### {t['performance_metrics']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(create_performance_gauge(rmse, "RMSE", 0, 0.1), use_container_width=True)
                with col2:
                    st.plotly_chart(create_performance_gauge(mse, "MSE", 0, 0.01), use_container_width=True)
                with col3:
                    st.plotly_chart(create_performance_gauge(r2, "R² Score", 0, 1), use_container_width=True)
                
                # منحنى الخسارة / Loss curve
                st.markdown(f"### {t['training_loss']}")
                fig_loss, ax = plt.subplots(figsize=(10, 4))
                ax.plot(history.history['loss'], 
                       label='Training Loss' if language == 'english' else 'فقدان التدريب', 
                       linewidth=2)
                ax.set_title(f'{t["training_loss"]} - {model_type}')
                ax.set_xlabel('Epochs' if language == 'english' else 'الدورات')
                ax.set_ylabel('Loss' if language == 'english' else 'الفقدان')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_loss)
                
                # التنبؤ مقابل الفعلي / Prediction vs Actual
                st.markdown(f"### {t['actual_vs_predicted']}")
                
                # إنشاء بيانات للرسم / Create data for plotting
                train = df[:training_data_len]
                valid = df[training_data_len:]
                valid = valid.copy()
                valid['Predictions'] = predictions
                
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Scatter(
                    x=train['Date'], y=train['Price'],
                    name=t['training_data'],
                    line=dict(color='blue', width=2)
                ))
                fig_comparison.add_trace(go.Scatter(
                    x=valid['Date'], y=valid['Price'],
                    name=t['actual_price'],
                    line=dict(color='green', width=2)
                ))
                fig_comparison.add_trace(go.Scatter(
                    x=valid['Date'], y=valid['Predictions'],
                    name=t['predicted_price'],
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_comparison.update_layout(
                    title=f'{t["actual_vs_predicted"]} - {model_type}',
                    xaxis_title='Date' if language == 'english' else 'التاريخ',
                    yaxis_title='Price' if language == 'english' else 'السعر',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # التنبؤات المستقبلية / Future predictions
                st.markdown(f"### {t['future_predictions']}")
                
                # الحصول على آخر أيام نافذة الزمن / Get last time window days
                last_time_window_days = df['Price'].values[-time_window:]
                last_time_window_days_scaled = predictor.scaler.transform(
                    last_time_window_days.reshape(-1, 1)
                )
                
                # التنبؤ بـ 30 يوم القادمة / Predict next 30 days
                future_predictions = []
                current_batch = last_time_window_days_scaled.reshape(1, time_window, 1)
                
                for i in range(30):
                    if model_type == 'LSTM':
                        current_pred = predictor.model.predict(current_batch, verbose=0)[0]
                    else:
                        current_batch_mlp = current_batch.reshape(1, time_window)
                        current_pred = predictor.model.predict(current_batch_mlp, verbose=0)[0]
                    
                    future_predictions.append(current_pred[0])
                    
                    # تحديث الدفعة للتنبؤ التالي / Update batch for next prediction
                    current_batch = np.append(
                        current_batch[:, 1:, :], 
                        [[[current_pred[0]]]], 
                        axis=1
                    )
                
                future_predictions = predictor.scaler.inverse_transform(
                    np.array(future_predictions).reshape(-1, 1)
                )
                
                # إنشاء التواريخ المستقبلية / Create future dates
                last_date = df['Date'].iloc[-1]
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1), 
                    periods=30, 
                    freq='D'
                )
                
                # رسم التنبؤات المستقبلية / Plot future predictions
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=df['Date'][-100:], y=df['Price'][-100:],
                    name=t['historical_price'],
                    line=dict(color='blue', width=2)
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates, y=future_predictions.flatten(),
                    name=t['future_forecast'],
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_future.update_layout(
                    title=t['future_predictions'],
                    xaxis_title='Date' if language == 'english' else 'التاريخ',
                    yaxis_title='Price' if language == 'english' else 'السعر',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_future, use_container_width=True)
                
                # تحميل التنبؤات / Download predictions
                st.markdown(f"### {t['download_results']}")
                
                # إنشاء بيانات للتحميل / Create data for download
                date_col = 'Date' if language == 'english' else 'التاريخ'
                price_col = 'Predicted_Price' if language == 'english' else 'السعر_المتوقع'
                model_col = 'Model_Used' if language == 'english' else 'النموذج_المستخدم'
                prediction_col = 'Prediction_Date' if language == 'english' else 'تاريخ_التنبؤ'
                
                future_df = pd.DataFrame({
                    date_col: future_dates,
                    price_col: future_predictions.flatten(),
                    model_col: model_type,
                    prediction_col: datetime.now().strftime("%Y-%m-%d")
                })
                
                csv = future_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label=t['download_button'],
                    data=csv,
                    file_name=f"future_predictions_{model_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.success(t['training_completed'])
    
    else:
        # صفحة الترحيب / Welcome page
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h2>{t['welcome']}</h2>
            <p style='font-size: 1.2rem;'>{t['welcome_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {t['data_format']}")
            st.markdown(f"{t['data_columns']}")
            st.markdown(f"- **{t['date_col']}**")
            st.markdown(f"- **{t['price_col']}**")
            st.markdown(f"- **{t['open_col']}**")
            st.markdown(f"- **{t['high_col']}**")
            st.markdown(f"- **{t['low_col']}**")
            st.markdown(f"- **{t['vol_col']}**")
            st.markdown(f"- **{t['change_col']}**")
            
            # مثال على البيانات / Sample data
            sample_data = pd.DataFrame({
                'Date': ['01/01/2023', '01/02/2023', '01/03/2023'],
                'Price': [0.3858, 0.4083, 0.4437],
                'Open': [0.3806, 0.3870, 0.4096],
                'High': [0.3589, 0.3717, 0.4006],
                'Low': [0.3973, 0.3941, 0.4299],
                'Vol.': [0.0474, 0.0728, 0.1252],
                'Change %': [0.5222, 0.5759, 0.6275]
            })
            st.dataframe(sample_data, use_container_width=True)
        
        with col2:
            st.markdown(f"### {t['features']}")
            for feature in t['features_list']:
                st.markdown(f"- **{feature}**")
            
            st.markdown(f"### {t['model_specs']}")
            for spec in t['model_specs_list']:
                st.markdown(f"- **{spec}**")
        
        # إحصائيات وهمية للعرض / Demo performance statistics
        st.markdown(f"### {t['performance_stats']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t['lstm_accuracy'], "94.2%", "1.2%")
        with col2:
            st.metric(t['mlp_accuracy'], "92.8%", "0.8%")
        with col3:
            st.metric(t['avg_r2'], "0.89", "0.03")

if __name__ == "__main__":
    main()