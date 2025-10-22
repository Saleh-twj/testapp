import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, Flatten
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
                "Deep Learning Models: LSTM, MLP, and Hybrid as per project document",
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
                "Hybrid: Combines LSTM temporal features with MLP pattern recognition",
                "Data Normalization: Min-Max scaling according to methodology",
                "Time Series: Sliding window approach for sequence prediction"
            ],
            'performance_stats': "📈 Model Performance Statistics",
            'lstm_accuracy': "LSTM Accuracy",
            'mlp_accuracy': "MLP Accuracy",
            'hybrid_accuracy': "Hybrid Accuracy",
            'avg_r2': "Average R²",
            'historical_price': "Historical Price",
            'future_forecast': "Future Forecast",
            'training_data': "Training Data",
            'actual_price': "Actual Price",
            'predicted_price': "Predicted Price",
            'hybrid_description': "🤖 Hybrid Model: Combines LSTM's sequence learning with MLP's pattern recognition for enhanced accuracy",
            'model_comparison': "📊 Model Comparison",
            'best_model': "🏆 Best Performing Model"
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
                "نماذج التعلم العميق: LSTM و MLP والهجين كما في وثيقة المشروع",
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
                "الهجين: يجمع بين ميزات LSTM الزمنية وتعلم الأنماط في MLP",
                "تطبيع البيانات: تحجيم Min-Max حسب المنهجية",
                "السلاسل الزمنية: نهج النافذة المنزلقة للتنبؤ بالتسلسل"
            ],
            'performance_stats': "📈 إحصائيات أداء النماذج",
            'lstm_accuracy': "دقة LSTM",
            'mlp_accuracy': "دقة MLP",
            'hybrid_accuracy': "دقة النموذج الهجين",
            'avg_r2': "متوسط R²",
            'historical_price': "السعر التاريخي",
            'future_forecast': "التنبؤات المستقبلية",
            'training_data': "بيانات التدريب",
            'actual_price': "السعر الفعلي",
            'predicted_price': "السعر المتوقع",
            'hybrid_description': "🤖 النموذج الهجين: يجمع بين تعلم التسلسل في LSTM والتعرف على الأنماط في MLP لتحسين الدقة",
            'model_comparison': "📊 مقارنة النماذج",
            'best_model': "🏆 أفضل نموذج أداء"
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
    .hybrid-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
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
    .model-comparison {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
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
    
    def build_hybrid_model(self, time_window, lstm_units=50, mlp_layers=[64, 32], dropout_rate=0.2):
        """بناء النموذج الهجين / Build Hybrid model"""
        # مدخل LSTM
        lstm_input = Input(shape=(time_window, 1), name='lstm_input')
        lstm_layer1 = LSTM(lstm_units, return_sequences=True)(lstm_input)
        lstm_dropout1 = Dropout(dropout_rate)(lstm_layer1)
        lstm_layer2 = LSTM(lstm_units, return_sequences=False)(lstm_dropout1)
        lstm_dropout2 = Dropout(dropout_rate)(lstm_layer2)
        lstm_output = Dense(25, activation='relu')(lstm_dropout2)
        
        # مدخل MLP (الميزات المسطحة)
        mlp_input = Input(shape=(time_window,), name='mlp_input')
        mlp_layer = Dense(mlp_layers[0], activation='relu')(mlp_input)
        
        for units in mlp_layers[1:]:
            mlp_layer = Dense(units, activation='relu')(mlp_layer)
        
        # دمج مخرجات LSTM و MLP
        combined = Concatenate()([lstm_output, mlp_layer])
        
        # طبقات إضافية بعد الدمج
        combined_layer = Dense(32, activation='relu')(combined)
        combined_layer = Dropout(dropout_rate)(combined_layer)
        combined_layer = Dense(16, activation='relu')(combined_layer)
        
        # طبقة الإخراج
        output_layer = Dense(1)(combined_layer)
        
        model = Model(inputs=[lstm_input, mlp_input], outputs=output_layer)
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
        elif model_type == 'MLP':
            # إعادة تشكيل البيانات لـ MLP / Reshape data for MLP
            x_train_mlp = x_train.reshape(x_train.shape[0], x_train.shape[1])
            self.model = self.build_mlp_model(time_window)
            history = self.model.fit(x_train_mlp, y_train, 
                                   batch_size=batch_size, 
                                   epochs=epochs,
                                   verbose=0)
        else:  # Hybrid
            self.model = self.build_hybrid_model(time_window)
            # تحضير البيانات للنموذج الهجين
            x_train_mlp = x_train.reshape(x_train.shape[0], x_train.shape[1])
            history = self.model.fit(
                [x_train, x_train_mlp], y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0
            )
        
        return history
    
    def predict(self, x_test, model_type='LSTM'):
        """إجراء التنبؤات / Make predictions"""
        if model_type == 'LSTM':
            predictions = self.model.predict(x_test, verbose=0)
        elif model_type == 'MLP':
            x_test_mlp = x_test.reshape(x_test.shape[0], x_test.shape[1])
            predictions = self.model.predict(x_test_mlp, verbose=0)
        else:  # Hybrid
            x_test_mlp = x_test.reshape(x_test.shape[0], x_test.shape[1])
            predictions = self.model.predict([x_test, x_test_mlp], verbose=0)
        
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

def create_model_comparison_chart(metrics_dict, lang):
    """إنشاء مخطط مقارنة النماذج / Create model comparison chart"""
    models = list(metrics_dict.keys())
    rmse_values = [metrics_dict[model]['rmse'] for model in models]
    r2_values = [metrics_dict[model]['r2'] for model in models]
    
    fig = go.Figure()
    
    # إضافة RMSE
    fig.add_trace(go.Bar(
        name='RMSE' if lang == 'english' else 'جذر متوسط مربع الخطأ',
        x=models,
        y=rmse_values,
        marker_color='indianred'
    ))
    
    # إضافة R²
    fig.add_trace(go.Bar(
        name='R² Score' if lang == 'english' else 'معدل التحديد R²',
        x=models,
        y=r2_values,
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        title='Model Comparison' if lang == 'english' else 'مقارنة النماذج',
        xaxis_title='Models' if lang == 'english' else 'النماذج',
        yaxis_title='Score' if lang == 'english' else 'القيمة',
        barmode='group',
        height=400
    )
    
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
        
        model_type = st.selectbox(t['select_model'], ["LSTM", "MLP", "Hybrid"])
        
        # Show hybrid model description
        if model_type == "Hybrid":
            st.info(t['hybrid_description'])
        
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
            
            # Train all models for comparison if hybrid is selected
            models_to_train = ["LSTM", "MLP", "Hybrid"] if model_type == "Hybrid" else [model_type]
            all_metrics = {}
            all_predictions = {}
            all_histories = {}
            
            for current_model in models_to_train:
                with st.spinner(f"{t['training_model']} ({current_model})"):
                    # تحضير البيانات / Prepare data
                    x_train, y_train, x_test, y_test, training_data_len = predictor.prepare_data(
                        df, time_window, test_ratio
                    )
                    
                    # تدريب النموذج / Train model
                    history = predictor.train_model(
                        x_train, y_train, current_model, epochs, batch_size, time_window
                    )
                    
                    # إجراء التنبؤات / Make predictions
                    predictions = predictor.predict(x_test, current_model)
                    
                    # حساب المقاييس / Calculate metrics
                    mse, rmse, r2 = predictor.calculate_metrics(y_test, predictions)
                    
                    # تخزين النتائج
                    all_metrics[current_model] = {'mse': mse, 'rmse': rmse, 'r2': r2}
                    all_predictions[current_model] = predictions
                    all_histories[current_model] = history
            
            # عرض مقارنة النماذج إذا كان الهجين / Show model comparison if hybrid
            if model_type == "Hybrid":
                st.markdown(f"### {t['model_comparison']}")
                
                # إنشاء مخطط المقارنة
                comparison_fig = create_model_comparison_chart(all_metrics, language)
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # تحديد أفضل نموذج
                best_model = min(all_metrics.keys(), key=lambda x: all_metrics[x]['rmse'])
                st.markdown(f"### {t['best_model']}: **{best_model}**")
                
                # عرض مقاييس جميع النماذج
                col1, col2, col3 = st.columns(3)
                models_display = ["LSTM", "MLP", "Hybrid"]
                colors = ["blue", "green", "orange"]
                
                for i, model in enumerate(models_display):
                    with [col1, col2, col3][i]:
                        if model in all_metrics:
                            metric_color = "orange" if model == best_model else colors[i]
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, {colors[i]} 0%, {colors[i]}80 100%); 
                                        padding: 1rem; border-radius: 10px; color: white; text-align: center;'>
                                <h4>{model}</h4>
                                <p><strong>RMSE:</strong> {all_metrics[model]['rmse']:.6f}</p>
                                <p><strong>R²:</strong> {all_metrics[model]['r2']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # استخدام أفضل نموذج للعروض التالية
                best_predictions = all_predictions[best_model]
                best_history = all_histories[best_model]
                display_model = best_model
            else:
                # استخدام النموذج المختار للعروض التالية
                best_predictions = all_predictions[model_type]
                best_history = all_histories[model_type]
                display_model = model_type
            
            # عرض النتائج / Display results
            st.markdown(f"### {t['performance_metrics']} - {display_model}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_performance_gauge(all_metrics[display_model]['rmse'], "RMSE", 0, 0.1), use_container_width=True)
            with col2:
                st.plotly_chart(create_performance_gauge(all_metrics[display_model]['mse'], "MSE", 0, 0.01), use_container_width=True)
            with col3:
                st.plotly_chart(create_performance_gauge(all_metrics[display_model]['r2'], "R² Score", 0, 1), use_container_width=True)
            
            # منحنى الخسارة / Loss curve
            st.markdown(f"### {t['training_loss']} - {display_model}")
            fig_loss, ax = plt.subplots(figsize=(10, 4))
            ax.plot(best_history.history['loss'], 
                   label='Training Loss' if language == 'english' else 'فقدان التدريب', 
                   linewidth=2)
            ax.set_title(f'{t["training_loss"]} - {display_model}')
            ax.set_xlabel('Epochs' if language == 'english' else 'الدورات')
            ax.set_ylabel('Loss' if language == 'english' else 'الفقدان')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
            
            # التنبؤ مقابل الفعلي / Prediction vs Actual
            st.markdown(f"### {t['actual_vs_predicted']} - {display_model}")
            
            # إنشاء بيانات للرسم / Create data for plotting
            train = df[:training_data_len]
            valid = df[training_data_len:]
            valid = valid.copy()
            valid['Predictions'] = best_predictions
            
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
                title=f'{t["actual_vs_predicted"]} - {display_model}',
                xaxis_title='Date' if language == 'english' else 'التاريخ',
                yaxis_title='Price' if language == 'english' else 'السعر',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # التنبؤات المستقبلية / Future predictions
            st.markdown(f"### {t['future_predictions']} - {display_model}")
            
            # استخدام النموذج الأفضل للتنبؤات المستقبلية
            predictor.model_type = display_model
            if display_model == "Hybrid":
                predictor.model = all_histories[display_model].model
            
            # الحصول على آخر أيام نافذة الزمن / Get last time window days
            last_time_window_days = df['Price'].values[-time_window:]
            last_time_window_days_scaled = predictor.scaler.transform(
                last_time_window_days.reshape(-1, 1)
            )
            
            # التنبؤ بـ 30 يوم القادمة / Predict next 30 days
            future_predictions = []
            current_batch = last_time_window_days_scaled.reshape(1, time_window, 1)
            
            for i in range(30):
                if display_model == 'LSTM':
                    current_pred = predictor.model.predict(current_batch, verbose=0)[0]
                elif display_model == 'MLP':
                    current_batch_mlp = current_batch.reshape(1, time_window)
                    current_pred = predictor.model.predict(current_batch_mlp, verbose=0)[0]
                else:  # Hybrid
                   