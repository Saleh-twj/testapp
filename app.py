import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ------------------------------------
# Language Setup
# ------------------------------------
LANGUAGES = {
    "English": {
        "title": "📈 Saudi Stock Prediction System",
        "select_stock": "Select a TASI Stock",
        "date_range": "Select Date Range",
        "predict_button": "Predict Future Prices",
        "learn_more": "Learn More",
        "learn_more_text": """
### About this Application
This system predicts future stock prices for companies listed on the Saudi Stock Exchange (TASI).
It uses **historical data** collected through Yahoo Finance and applies **machine learning (LSTM)** models
to forecast short-term movements.

#### 🔍 How it Works:
1. Select a company (stock ticker) from the dropdown.
2. The app downloads the past 5 years of data from Yahoo Finance.
3. Data is normalized using MinMaxScaler.
4. The LSTM model is trained and predicts the next 30 days.
5. The predictions are visualized interactively.

#### ⚙️ Notes:
- Predictions are **experimental** and for **educational purposes only**.
- Not for financial trading decisions.
        """,
        "prediction_results": "Prediction Results",
        "model_metrics": "Model Performance Metrics",
        "rmse": "RMSE (Root Mean Square Error)",
        "r2": "R² Score",
        "future_forecast": "📉 Future 30-Day Forecast",
        "download_excel": "Download Predictions as Excel File",
        "download_report": "📥 Download Report",
        "report_preview": "📄 Report Preview",
        "combined_chart": "📊 Combined Historical and Forecasted Data",
        "success": "✅ Prediction completed successfully!",
        "disclaimer": "⚠️ Disclaimer: Educational use only. Not financial advice.",
        "info": "👆 Select a stock and click 'Predict Future Prices' to begin.",
        "training_message": "⏳ Training LSTM model... please wait.",
        "file_name_excel": "Predictions.xlsx",
        "file_name_report": "Report.txt",
        "report_title": "Stock Prediction Report",
        "report_stock": "Stock",
        "report_date": "Date",
        "report_metrics": "Model Metrics",
        "report_notes": "Notes",
        "report_note_text": "- Predictions are based on recent market data and ML techniques.\n- Use for educational and research purposes only.",
    },
    "العربية": {
        "title": "📈 نظام التنبؤ بأسعار الأسهم السعودية",
        "select_stock": "اختر سهماً من السوق السعودي (تاسي)",
        "date_range": "حدد النطاق الزمني",
        "predict_button": "تنبؤ بالأسعار المستقبلية",
        "learn_more": "معلومات تفصيلية",
        "learn_more_text": """
### عن هذا النظام
يقوم هذا النظام بالتنبؤ بأسعار الأسهم المستقبلية للشركات المدرجة في **السوق المالية السعودية (تاسي)**.
ويعتمد على **البيانات التاريخية** من موقع Yahoo Finance باستخدام خوارزمية **التعلم العميق (LSTM)**.

#### 🔍 آلية العمل:
1. اختر الشركة (رمز السهم) من القائمة.
2. يتم تحميل بيانات آخر 5 سنوات.
3. يتم تطبيع البيانات باستخدام MinMaxScaler.
4. يتم تدريب نموذج LSTM لتنبؤ الثلاثين يوماً القادمة.
5. تُعرض النتائج بطريقة تفاعلية.

#### ⚙️ ملاحظات:
- هذه التنبؤات **تجريبية** وللأغراض **التعليمية فقط**.
- لا يُنصح بالاعتماد عليها في قرارات الاستثمار.
        """,
        "prediction_results": "نتائج التنبؤ",
        "model_metrics": "مقاييس أداء النموذج",
        "rmse": "متوسط الجذر التربيعي للخطأ (RMSE)",
        "r2": "معامل التحديد (R²)",
        "future_forecast": "📉 التنبؤ لثلاثين يوماً قادمة",
        "download_excel": "تحميل التنبؤات كملف Excel",
        "download_report": "📥 تحميل التقرير",
        "report_preview": "📄 معاينة التقرير",
        "combined_chart": "📊 البيانات التاريخية والتنبؤات المستقبلية",
        "success": "✅ تم إكمال عملية التنبؤ بنجاح!",
        "disclaimer": "⚠️ تنويه: للاستخدام التعليمي فقط، وليست نصيحة مالية.",
        "info": "👆 اختر سهماً واضغط على 'تنبؤ بالأسعار المستقبلية' للبدء.",
        "training_message": "⏳ جاري تدريب نموذج LSTM... الرجاء الانتظار.",
        "file_name_excel": "التنبؤات.xlsx",
        "file_name_report": "التقرير.txt",
        "report_title": "تقرير التنبؤ بالسهم",
        "report_stock": "السهم",
        "report_date": "التاريخ",
        "report_metrics": "مقاييس النموذج",
        "report_notes": "ملاحظات",
        "report_note_text": "- التنبؤات مبنية على بيانات السوق وتقنيات تعلم الآلة.\n- للاستخدام التعليمي والبحثي فقط.",
    },
}

# ------------------------------------
# Streamlit Setup
# ------------------------------------
st.set_page_config(page_title="Stock Prediction", layout="wide")

language = st.sidebar.selectbox("Language / اللغة", list(LANGUAGES.keys()))
L = LANGUAGES[language]

st.title(L["title"])

with st.expander(L["learn_more"]):
    st.markdown(L["learn_more_text"])

# ------------------------------------
# Stock Options
# ------------------------------------
tasi_stocks = {
    "Al Rajhi Bank": "1120.SR",
    "SABIC": "2010.SR",
    "Saudi Aramco": "2222.SR",
    "STC": "7010.SR",
    "Riyad Bank": "1010.SR",
    "Alinma Bank": "1150.SR",
}

selected_stock = st.selectbox(L["select_stock"], list(tasi_stocks.keys()))
ticker = tasi_stocks[selected_stock]

end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)
st.date_input(L["date_range"], (start_date, end_date))

# ------------------------------------
# Load Data
# ------------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker)

st.subheader("📊 Historical Data")
st.line_chart(data["Close"])

# ------------------------------------
# Prediction Logic
# ------------------------------------
if st.button(L["predict_button"]):
    st.info(L["training_message"])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    training_data_len = int(len(scaled_data) * 0.8)

    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]

    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

    # Test
    X_test, y_test = [], scaled_data[training_data_len:]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(mean_squared_error(data["Close"][training_data_len:], predictions))
    r2 = r2_score(data["Close"][training_data_len:], predictions)

    # Metrics
    st.subheader(L["model_metrics"])
    st.write(f"{L['rmse']}: {rmse:.2f}")
    st.write(f"{L['r2']}: {r2:.2f}")

    # Chart
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid["Predictions"] = predictions

    fig = px.line()
    fig.add_scatter(x=train["Date"], y=train["Close"], name="Training Data")
    fig.add_scatter(x=valid["Date"], y=valid["Close"], name="Actual Price")
    fig.add_scatter(x=valid["Date"], y=valid["Predictions"], name="Predicted Price")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast
    last_60_days = scaled_data[-60:]
    X_future = np.array([last_60_days])
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

    future_predictions = []
    for _ in range(30):
        pred = model.predict(X_future)
        future_predictions.append(pred[0, 0])
        X_future = np.append(X_future[:, 1:, :], [[pred]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(data["Date"].iloc[-1], periods=30, freq="D")
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})

    st.subheader(L["future_forecast"])
    st.line_chart(future_df.set_index("Date"))

    # Export Options
    col1, col2, col3 = st.columns(3)

    with col1:
        future_df.to_excel(L["file_name_excel"], index=False)
        with open(L["file_name_excel"], "rb") as f:
            st.download_button(
                label=L["download_excel"],
                data=f,
                file_name=f"{selected_stock}_{L['file_name_excel']}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col2:
        st.dataframe(future_df.style.format({"Predicted Price": "{:.2f}"}))

    with col3:
        report = f"""
{L['report_title']}
-----------------------
{L['report_stock']}: {selected_stock} ({ticker})
{L['report_date']}: {datetime.today().strftime('%Y-%m-%d')}

{L['report_metrics']}:
RMSE: {rmse:.2f}
R²: {r2:.2f}

{L['report_notes']}:
{L['report_note_text']}
"""
        st.text_area(L["report_preview"], report, height=180)
        st.download_button(
            label=L["download_report"],
            data=report.encode("utf-8"),
            file_name=f"{selected_stock}_{L['file_name_report']}",
            mime="text/plain",
        )

    st.subheader(L["combined_chart"])
    combined_df = pd.concat([
        data[["Date", "Close"]].rename(columns={"Close": "Price"}).tail(120),
        future_df.rename(columns={"Predicted Price": "Price"})
    ])
    fig2 = px.line(combined_df, x="Date", y="Price", labels={"Price": "Stock Price"})
    st.plotly_chart(fig2, use_container_width=True)

    st.success(L["success"])
    st.caption(L["disclaimer"])

else:
    st.info(L["info"])
