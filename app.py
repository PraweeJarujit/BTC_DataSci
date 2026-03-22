import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as st_sns # ใช้ชื่อ st_sns กันสับสน

# ==========================================
# 1. ตั้งค่าหน้าเพจและ ธีม (Page Config)
# ==========================================
st.set_page_config(
    page_title="Bitcoin AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ฟังก์ชันโหลดโมเดล (Cache ไว้จะได้ไม่อืด)
# ==========================================
@st.cache_resource
def load_model():
    # โหลดไฟล์ pkl (เช็คชื่อไฟล์ให้ตรงกับที่คุณเซฟมานะครับ)
    try:
        model = joblib.load('best_bitcoin_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล 'best_bitcoin_model.pkl' กรุณาตรวจสอบว่าวางไฟล์ไว้ในโฟลเดอร์เดียวกันหรือไม่")
        return None

model = load_model()

# ==========================================
# 3. ออกแบบแถบด้านข้าง (Sidebar) สำหรับรับ Input
# ==========================================
st.sidebar.header("⚙️ ปรับแต่งค่า Technical Indicators")
st.sidebar.markdown("กรอกข้อมูลสถานะตลาดปัจจุบัน เพื่อให้ AI ทำนายทิศทางราคาพรุ่งนี้")

# สร้าง Slider และ Number Input สำหรับ 7 Features
daily_return = st.sidebar.number_input("Daily Return (%)", value=0.50, step=0.1) / 100
sma_50 = st.sidebar.number_input("SMA 50 Days (USD)", value=60000.0, step=100.0)
sma_200 = st.sidebar.number_input("SMA 200 Days (USD)", value=55000.0, step=100.0)
rsi_14 = st.sidebar.slider("RSI 14 Days", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
macd = st.sidebar.number_input("MACD", value=150.0, step=10.0)
volatility = st.sidebar.slider("Volatility (High-Low/Open)", min_value=0.0, max_value=0.20, value=0.03, step=0.001)
volume = st.sidebar.number_input("Trading Volume", value=35000.0, step=1000.0)

# ==========================================
# 4. ออกแบบหน้าจอหลัก (Main Dashboard)
# ==========================================
st.title("📈 Bitcoin Price Trend Predictor (AI Assistant)")
st.markdown("""
แอปพลิเคชันนี้ใช้โมเดล Machine Learning (XGBoost) ในการวิเคราะห์ Technical Indicators 
เพื่อทำนายว่า **ทิศทางราคาบิตคอยน์ในวันพรุ่งนี้จะ "ขึ้น" หรือ "ลง"** พร้อมระบุระดับความมั่นใจ (Probability) 
เพื่อช่วยนักลงทุนประกอบการตัดสินใจและบริหารความเสี่ยง
""")

st.divider()

# รวมข้อมูล Input เข้าเป็น DataFrame (ชื่อคอลัมน์ต้องเป๊ะเหมือนตอนเทรน)
input_data = pd.DataFrame({
    'Daily_Return': [daily_return],
    'SMA_50': [sma_50],
    'SMA_200': [sma_200],
    'RSI_14': [rsi_14],
    'MACD': [macd],
    'Volatility': [volatility],
    'Volume': [volume]
})

if model is not None:
    # ------------------------------------------
    # ส่วนที่ 1: การทำนายผล (Prediction & Probability)
    # ------------------------------------------
    st.subheader("🤖 ผลการทำนายจาก AI (Prediction)")
    
    # สั่งให้โมเดลทำนาย
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # แยกความน่าจะเป็นของแต่ละคลาส
    prob_down = probabilities[0] * 100
    prob_up = probabilities[1] * 100

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ทิศทางราคาพรุ่งนี้:")
        if prediction == 1:
            st.success("🟢 มีแนวโน้มปรับตัวสูงขึ้น (UP)")
            st.metric(label="โอกาสราคาขึ้น (Confidence)", value=f"{prob_up:.2f}%")
            st.progress(prob_up / 100)
            st.markdown("**กลยุทธ์แนะนำ:** พิจารณาหาจังหวะเข้าซื้อ (Long) หรือถือครอง (Hold) ต่อไป")
        else:
            st.error("🔴 มีแนวโน้มปรับตัวลดลง (DOWN)")
            st.metric(label="โอกาสราคาลง (Confidence)", value=f"{prob_down:.2f}%")
            st.progress(prob_down / 100)
            st.markdown("**กลยุทธ์แนะนำ:** พิจารณาชะลอการซื้อ, ทยอยขายทำกำไร หรือตั้ง Stop-loss อย่างระมัดระวัง")

    with col2:
        # แสดงตารางข้อมูลที่ผู้ใช้กรอกเข้ามา
        st.markdown("### ข้อมูลที่ระบบใช้วิเคราะห์:")
        st.dataframe(input_data.style.format("{:.4f}"), hide_index=True)

    st.divider()

    # ------------------------------------------
    # ส่วนที่ 2: กราฟโบนัส Feature Importance (แสดงความโปร)
    # ------------------------------------------
    st.subheader("📊 ปัจจัยที่มีผลต่อการตัดสินใจของ AI (Feature Importance)")
    st.markdown("กราฟนี้แสดงให้เห็นว่า AI ให้น้ำหนักกับตัวแปรใดมากที่สุดในการทำนายผลครั้งนี้ (เรียงจากมากไปน้อย)")

    try:
        # ดึงโมเดล XGBoost ออกมาจาก Pipeline
        xgb_model = model.named_steps['classifier']
        importances = xgb_model.feature_importances_
        features_list = input_data.columns

        # สร้าง DataFrame สำหรับพล็อต
        feat_df = pd.DataFrame({'Feature': features_list, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)

        # วาดกราฟ
        fig, ax = plt.subplots(figsize=(10, 4))
        # ใช้สีสไตล์ Hacker/Cyberpunk
        st_sns.barplot(x='Importance', y='Feature', data=feat_df, palette='mako', ax=ax)
        ax.set_title('Feature Importance from XGBoost', fontsize=14)
        ax.set_xlabel('Relative Importance')
        ax.set_ylabel('')
        
        # แสดงกราฟบน Streamlit
        st.pyplot(fig)
        
    except Exception as e:
        st.warning("โมเดลไม่รองรับการแสดง Feature Importance หรือโครงสร้าง Pipeline ไม่ตรงกัน")

else:
    st.warning("รอการเชื่อมต่อโมเดล...")

# ==========================================
# 5. ส่วนอธิบายท้ายเพจ
# ==========================================
with st.expander("ℹ️ ข้อควรระวังและการแปลผล (Disclaimer)"):
    st.write("""
    - โมเดลนี้ถูกฝึกสอนมาโดยมุ่งเน้นที่ค่า **Precision** เพื่อลดสัญญาณหลอก (False Positive) 
    - ความน่าจะเป็น (Probability) คือระดับความมั่นใจของตัวโมเดลอิงจากสถิติในอดีต ไม่ใช่การการันตีอนาคต
    - การลงทุนมีความเสี่ยง ควรใช้ AI เป็นเพียงหนึ่งในเครื่องมือช่วยตัดสินใจ (Trading Assistant) ไม่ใช่ตัวตัดสินใจหลัก
    """)