import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as st_sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
# 1.1 ฟังก์ชันตรวจสอบหน้าจอ (Mobile Detection)
# ==========================================
def is_mobile():
    """ตรวจสอบว่าเป็นอุปกรณ์พกพาหรือไม่"""
    try:
        # แก้ไขสำหรับ Streamlit เวอร์ชันใหม่ (ลบวงเล็บหลัง query_params ออก)
        return st.query_params.get("mobile", "false") == "true" or st.session_state.get("force_mobile", False)
    except:
        return False

# ==========================================
# 1.2 บังคับใช้ Dark Mode เท่านั้น
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
.stButton>button {
    background-color: #1f2833;
    color: #66fcf1;
    border: 2px solid #66fcf1;
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #66fcf1;
    color: #0b0c10;
    box-shadow: 0 0 10px #66fcf1;
    transform: translateY(-2px);
}
.stSelectbox>div>div>select {
    background-color: #262730;
    color: #FAFAFA;
    padding: 12px;
    border-radius: 8px;
}
.stSlider>div>div>div {
    background-color: #262730;
    border-radius: 8px;
}
.stNumberInput>div>div>input {
    background-color: #262730;
    color: #FAFAFA;
    padding: 12px;
    border-radius: 8px;
}
.stDataFrame {
    background-color: #262730;
    color: #FAFAFA;
    border-radius: 8px;
}
.stAlert {
    background-color: #262730;
    color: #FAFAFA;
    border-radius: 8px;
}
.stProgress>div>div>div>div {
    background-color: #262730;
    border-radius: 8px;
}
.stMetric {
    background-color: #262730;
    color: #FAFAFA;
    border-radius: 8px;
    padding: 16px;
}
/* Mobile Responsive */
@media (max-width: 768px) {
    .stButton>button {
        padding: 16px 20px;
        font-size: 18px;
        min-height: 48px;
    }
    .stSelectbox>div>div>select {
        padding: 16px;
        font-size: 16px;
        min-height: 48px;
    }
    .stNumberInput>div>div>input {
        padding: 16px;
        font-size: 16px;
        min-height: 48px;
    }
    .stSlider>div>div>div {
        min-height: 48px;
    }
}
</style>
""", unsafe_allow_html=True)

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
# 3. ฟังก์ชันตรวจสอบค่า Input (Input Validation)
# ==========================================
def validate_inputs(daily_return, sma_50, sma_200, rsi_14, macd, volatility, volume):
    """ตรวจสอบค่าที่ผู้ใช้กรอกว่าสมเหตุสมผลหรือไม่"""
    errors = []
    
    # Daily Return ต้องอยู่ในช่วง -20% ถึง +20%
    if abs(daily_return) > 0.20:
        errors.append("Daily Return ไม่ควรเกิน ±20% (เปลี่ยนแปลงมากกว่านี้ผิดปกติ)")
    
    # SMA 50 ไม่จำเป็นต้องมากกว่า SMA 200 (Death Cross เป็นสัญญาณลงที่ถูกต้อง)
    
    # SMA ต้องอยู่ในช่วงที่เป็นเหตุผล (1,000 - 200,000 USD)
    if not (1000 <= sma_50 <= 200000):
        errors.append("SMA 50 ควรอยู่ในช่วง 1,000 - 200,000 USD")
    if not (1000 <= sma_200 <= 200000):
        errors.append("SMA 200 ควรอยู่ในช่วง 1,000 - 200,000 USD")
    
    # RSI ต้องอยู่ในช่วง 0-100 (slider กำหนดแล้ว)
    # MACD ไม่ควรเกิน ±5,000
    if abs(macd) > 5000:
        errors.append("MACD ไม่ควรเกิน ±5,000")
    
    # Volatility ต้องอยู่ในช่วง 0-1 (slider กำหนดแล้ว)
    # Volume ต้องเป็นบวกและไม่เกิน 1M
    if volume <= 0:
        errors.append("Trading Volume ต้องมากกว่า 0")
    if volume > 1000000:
        errors.append("Trading Volume ไม่ควรเกิน 1,000,000")
    
    return errors

# ==========================================
# 4. ออกแบบแถบด้านข้าง (Sidebar) สำหรับข้อมูลช่วยเหลือ
# ==========================================
# สร้าง Slider และ Number Input สำหรับ 7 Features
with st.sidebar.expander("📖 คำอธิบาย Technical Indicators"):
    st.markdown("""
    **📊 Daily Return (%)**: การเปลี่ยนแปลงราคาในวันนี้ (บวก=ราคาขึ้น, ลบ=ราคาลง)
    
    **📈 SMA 50/200 Days**: ราคาเฉลี่ยเคลื่อนที่ 50/200 วัน บอกแนวโน้มระยะกลาง/ยาว
    
    **🔄 RSI 14 Days**: ดัชนีแรงซื้อแรงขาย 0-100 (70+=ซื้อเกิน, 30-=ขายเกิน)
    
    **⚡ MACD**: ความเร็วในการเปลี่ยนแปลงของราคา (บวก=ขึ้นเร็ว, ลบ=ลงเร็ว)
    
    **📊 Volatility**: ความผันผวนของราคา (สูง=ผันผวนมาก, ต่ำ=นิ่ง)
    
    **💰 Trading Volume**: ปริมาณการซื้อขาย (สูง=คนซื้อขายเยอะ, ต่ำ=ซื้อขายน้อย)
    """)

with st.sidebar.expander("⚠️ ข้อควรระวังที่สำคัญ (Disclaimer)"):
    st.error("""
    ## 🚨 คำเตือนสำคัญก่อนใช้งาน
    
    **❌ สิ่งที่แอปนี้ไม่ใช่:**
    - ❌ ไม่ใช่การการันตีว่าจะได้กำไร 100%
    - ❌ ไม่ใช่คำแนะนำการลงทุนที่ต้องตามอย่างเคร่งครัด
    - ❌ ไม่ใช่เครื่องมือทำนายอนาคตที่แม่นยำเสมอไป
    
    **✅ สิ่งที่แอปนี้คือ:**
    - ✅ เครื่องมือช่วยวิเคราะห์ข้อมูลตลาด
    - ✅ ข้อมูลเพิ่มเติมสำหรับประกอบการตัดสินใจ
    - ✅ แนวทางในการบริหารความเสี่ยง
    
    ## 📊 ความน่าจะเป็น (Probability) คืออะไร?
    - ค่าเปอร์เซ็นต์ที่แสดงคือ **ความมั่นใจของ AI** อิงจากสถิติในอดีต
    - เช่น 85% หมายถึง AI มั่นใจ 85% ว่าจะเป็นไปตามที่ทาย
    - **ไม่ใช่** การการันตีว่าจะถูกต้อง 85%
    
    ## 🎯 หลักการใช้งานอย่างปลอดภัย:
    1. **ใช้เป็นข้อมูลเสริม** อย่าพึ่งพาอย่างเดียว
    2. **ตั้ง Stop-loss** ทุกครั้งที่เทรด
    3. **ลงทุนเท่าที่สามารถขาดทุนได้**
    4. **ศึกษาข้อมูลจากแหล่งอื่น** ประกอบด้วย
    5. **ไม่ลงทุนเมื่อไม่เข้าใจ** ความเสี่ยงที่เผชิญ
    
    ## 📈 ความแม่นยำของโมเดล:
    - โมเดลนี้ถูกฝึกเพื่อเน้นค่า **Precision** (ลดสัญญาณหลอก)
    - ยังมีโอกาสทำนายผิดได้ โดยเฉพาะในตลาดผันผวนสูง
    - ผลการทำนายอาจไม่เหมือนจริงในสถานการณ์พิเศษ (ข่าวใหญ่, เหตุการณ์ไม่คาดคิด)
    
    ---
    **⚠️ การลงทุนมีความเสี่ยง ผู้ลงทุนควรศึกษาข้อมูลก่อนตัดสินใจ**
    """)

# ==========================================
# 5. ออกแบบหน้าจอหลัก (Main Dashboard)
# ==========================================
st.title("📈 ทำนายราคาบิตคอยน์พรุ่งนี้ (AI Assistant)")
st.markdown("""
### 🤖 AI ช่วยทำนายทิศทางราคาบิตคอยน์

แอปนี้ใช้ **ปัญญาประดิษฐ์ (AI)** วิเคราะห์ข้อมูลตลาด 7 ประเภท เพื่อทายว่า **ราคาบิตคอยน์วันพรุ่งนี้จะขึ้นหรือลง** พร้อมบอกความมั่นใจในการทายว่ากี่เปอร์เซ็นต์

**🎯 เหมาะสำหรับ:** นักลงทุนที่ต้องการข้อมูลเพิ่มเติมในการตัดสินใจ
**⚠️ ไม่ใช่:** การการันตีผลกำไร หรือคำแนะนำการลงทุนที่ต้องตาม
""")

st.divider()

# ==========================================
# 4.1 ส่วนรับข้อมูลจากผู้ใช้ (Input Section)
# ==========================================
st.subheader("📊 ขั้นตอนที่ 1: กรอกข้อมูลตลาดปัจจุบัน (Step 1: Enter Market Data)")
st.markdown("กรอกข้อมูลสถานะตลาดปัจจุบัน เพื่อให้ AI ทำนายทิศทางราคาพรุ่งนี้")

# สร้าง Grid Layout สำหรับ Input Widgets
col1, col2, col3 = st.columns(3)

with col1:
    daily_return = st.number_input("Daily Return (%)", value=0.50, step=0.1, format="%.2f", help="การเปลี่ยนแปลงราคาในวันนี้ (เช่น +2.5 หรือ -1.8)") / 100
    sma_50 = st.number_input("SMA 50 Days (USD)", value=60000.0, step=100.0, format="%.0f", help="ราคาเฉลี่ย 50 วันล่าสุด")
    volatility = st.slider("Volatility (High-Low/Open)", min_value=0.0, max_value=0.20, value=0.03, step=0.001, format="%.3f", help="ความผันผวนของราคา (0.01=นิ่ง, 0.10=ผันผวนมาก)")

with col2:
    sma_200 = st.number_input("SMA 200 Days (USD)", value=55000.0, step=100.0, format="%.0f", help="ราคาเฉลี่ย 200 วันล่าสุด")
    rsi_14 = st.slider("RSI 14 Days", min_value=0.0, max_value=100.0, value=55.0, step=1.0, help="ดัชนีแรงซื้อแรงขาย (30=ขายเกิน, 70=ซื้อเกิน)")
    volume = st.number_input("Trading Volume", value=35000.0, step=1000.0, format="%.0f", help="ปริมาณการซื้อขาย (หน่วย: BTC)")

with col3:
    macd = st.number_input("MACD", value=150.0, step=10.0, format="%.2f", help="ความเร็วการเปลี่ยนแปลงราคา (บวก=ขึ้นเร็ว)")

# แสดงสถานะการตรวจสอบค่า input (แสดงหลังจาก grid layout)
st.markdown("**สถานะการตรวจสอบ:**")
errors = validate_inputs(daily_return, sma_50, sma_200, rsi_14, macd, volatility, volume)

if errors:
    st.error("⚠️ พบข้อผิดพลาด:")
    for error in errors:
        st.error(f"• {error}")
    st.warning("กรุณาแก้ไขค่าให้ถูกต้องก่อนทำนาย")
    input_valid = False
else:
    st.success("✅ ค่าที่กรอกสมเหตุสมผล")
    input_valid = True

st.divider()

# ==========================================
# 4.2 ปุ่มทำนาย (CTA Button)
# ==========================================
st.subheader("🎯 ขั้นตอนที่ 2: ให้ AI วิเคราะห์ทิศทาง")

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

if model is not None and input_valid:
    # ------------------------------------------
    # ส่วนที่ 1: การทำนายผล (Prediction & Probability)
    # ------------------------------------------
    st.subheader("🤖 ผลการทำนายจาก AI")
    st.markdown("**AI วิเคราะห์ข้อมูลตลาดแล้ว ทายว่าราคาพรุ่งนี้จะ...**")
    
    # ปุ่มทำนายพร้อม loading animation
    if st.button("🚀 เริ่มทำนายผลทันที", use_container_width=True, type="primary"):
        # แสดง loading animation
        with st.spinner("🤖 AI กำลังวิเคราะห์ข้อมูล..."):
            # จำลองการทำงาน 2 วินาที
            import time
            time.sleep(2)
            
            # สั่งให้โมเดลทำนาย
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # แยกความน่าจะเป็นของแต่ละคลาส
            prob_down = probabilities[0] * 100
            prob_up = probabilities[1] * 100
            
            # แสดง success feedback พร้อม animation ที่เหมาะสมกับสถานะตลาด
            if prediction == 1:
                st.toast("✅ การทำนายเสร็จสมบูรณ์! (ตลาดกระทิง)", icon="🚀")
                st.balloons()  # ฉลองฉลองสำหรับตลาดขาขึ้น
            else:
                st.toast("✅ การทำนายเสร็จสมบูรณ์! (ฤดูหนาวคริปโต)", icon="❄️")
                st.snow()  # หิมะสำหรับตลาดขาลง (crypto winter)
            
            # เก็บผลลัพธ์ไว้ใน session state
            st.session_state.prediction_result = {
                'prediction': prediction,
                'prob_down': prob_down,
                'prob_up': prob_up
            }
            st.rerun()
    
    # แสดงผลลัพธ์ถ้ามีใน session state
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        prediction = result['prediction']
        prob_down = result['prob_down']
        prob_up = result['prob_up']
    else:
        # ถ้ายังไม่มีผลลัพธ์ ให้แสดงข้อความแทน
        st.info("👆 กดปุ่ม 'เริ่มทำนาย' เพื่อดูผลลัพธ์")
        prediction = None
        prob_down = 0
        prob_up = 0

    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### ทิศทางราคาพรุ่งนี้:")
        if prediction is not None:
            if prediction == 1:
                st.success("🟢 **ราคามีแนวโน้มขึ้น**")
                st.metric(label="📊 ความมั่นใจของ AI", value=f"{prob_up:.2f}%")
                st.progress(float(prob_up / 100))
                st.markdown("""**💡 แนะนำ:** พิจารณาหาจังหวะเข้าซื้อ หรือถือต่อถ้ามีอยู่แล้ว
- อาจเหมาะสำหรับการเปิด Long Position
- ควรตั้ง Stop-loss เพื่อความปลอดภัย""")
            else:
                st.error("🔴 **ราคามีแนวโน้มลง**")
                st.metric(label="📊 ความมั่นใจของ AI", value=f"{float(prob_down):.2f}%")
                st.progress(float(prob_down / 100))
                st.markdown("""**💡 แนะนำ:** ระมัดระวังการซื้อในช่วงนี้
- อาจเหมาะสำหรับการทยอยขายทำกำไร ถ้ามีอยู่
- ควรตั้ง Stop-loss ใกล้ราคาปัจจุบัน
- รอจังหวะราคาลงไปอีกก่อนตัดสินใจ""")
        else:
            st.info("⏳ รอการทำนาย...")
    
    with col2:
        if prediction is not None:
            # Gauge Chart สำหรับ Confidence Level
            confidence = prob_up if prediction == 1 else prob_down
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "ระดับความมั่นใจ"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, width='stretch')
        else:
            st.info("📊 รอการทำนาย...")
        
    with col3:
        if prediction is not None:
            # Risk Level Indicator
            confidence = prob_up if prediction == 1 else prob_down
            if confidence >= 80:
                risk_level = "ต่ำ"
                risk_color = "🟢"
                risk_emoji = "😌"
            elif confidence >= 60:
                risk_level = "ปานกลาง"
                risk_color = "🟡"
                risk_emoji = "🤔"
            else:
                risk_level = "สูง"
                risk_color = "🔴"
                risk_emoji = "😰"
                
            st.markdown(f"### {risk_emoji} ระดับความเสี่ยง")
            st.markdown(f"### {risk_color} {risk_level}")
            
            # คำอธิบายความเสี่ยง
            if confidence >= 80:
                st.info("AI มั่นใจสูง\nความเสี่ยงต่ำ")
            elif confidence >= 60:
                st.warning("AI มั่นใจปานกลาง\nความเสี่ยงปานกลาง")
            else:
                st.error("AI มั่นใจต่ำ\nความเสี่ยงสูง")
                
            st.caption("📊 อิงจากระดับความมั่นใจของ AI")
        else:
            st.info("⚠️ รอการทำนาย...")

    st.divider()
    
    # ------------------------------------------
    # ส่วนที่ 2: ข้อมูลที่ AI ใช้วิเคราะห์
    # ------------------------------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 ข้อมูลที่ AI ใช้วิเคราะห์:")
        st.dataframe(input_data.style.format("{:.4f}"), hide_index=True)
        st.caption("📊 ตารางนี้แสดงค่าที่คุณกรอก ซึ่ง AI จะนำไปวิเคราะห์")
        
    with col2:
        # สร้าง Market Condition Summary
        st.markdown("### 📈 สรุปสภาพตลาด")
        
        # วิเคราะห์สภาพตลาดจาก indicators
        market_analysis = []
        
        if rsi_14 > 70:
            market_analysis.append("🔴 ซื้อเกิน (Overbought)")
        elif rsi_14 < 30:
            market_analysis.append("🟢 ขายเกิน (Oversold)")
        else:
            market_analysis.append("🟡 ปกติ (Neutral)")
            
        if sma_50 > sma_200:
            market_analysis.append("📈 แนวโน้มขาขึ้น (Uptrend)")
        else:
            market_analysis.append("📉 แนวโน้มขาลง (Downtrend)")
            
        if volatility > 0.05:
            market_analysis.append("⚡ ผันผวนสูง (High Volatility)")
        elif volatility < 0.02:
            market_analysis.append("😴 ผันผวนต่ำ (Low Volatility)")
        else:
            market_analysis.append("📊 ผันผวนปานกลาง (Medium Volatility)")
            
        if macd > 0:
            market_analysis.append("🚀 โมเมนตัมบวก (Positive Momentum)")
        else:
            market_analysis.append("🛑 โมเมนตัมลบ (Negative Momentum)")
            
        # แสดงผลการวิเคราะห์
        for analysis in market_analysis:
            st.markdown(f"- {analysis}")
            
        # สรุปภาพรวม
        if len([a for a in market_analysis if "🟢" in a or "📈" in a or "🚀" in a]) >= 3:
            st.success("🎯 ภาพรวม: สัญญาณบวกเด่นชัด")
        elif len([a for a in market_analysis if "🔴" in a or "📉" in a or "🛑" in a]) >= 3:
            st.error("⚠️ ภาพรวม: สัญญาณลบเด่นชัด")
        else:
            st.warning("🔄 ภาพรวม: สัญญาณขัดแย้งกัน (Mixed Signals)")

    # ------------------------------------------
    # ส่วนที่ 3: กราฟโบนัส Feature Importance
    # ------------------------------------------
    st.subheader("📊 ปัจจัยสำคัญที่ AI พิจารณา")
    st.markdown("**กราฟนี้บอกว่า AI ให้ความสำคัญกับข้อมูลตัวไหนมากที่สุด** (เรียงจากมากไปน้อย)")

    try:
        # ดึงโมเดล XGBoost ออกมาจาก Pipeline
        xgb_model = model.named_steps['classifier']
        importances = xgb_model.feature_importances_
        features_list = input_data.columns

        # สร้าง DataFrame สำหรับพล็อต
        feat_df = pd.DataFrame({'Feature': features_list, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        # เพิ่มชื่อภาษาไทยสำหรับแสดงผล
        feature_names_thai = {
            'Daily_Return': 'การเปลี่ยนแปลงราคาวันนี้',
            'SMA_50': 'ราคาเฉลี่ย 50 วัน',
            'SMA_200': 'ราคาเฉลี่ย 200 วัน',
            'RSI_14': 'ดัชนีแรงซื้อแรงขาย',
            'MACD': 'ความเร็วการเปลี่ยนแปลง',
            'Volatility': 'ความผันผวน',
            'Volume': 'ปริมาณการซื้อขาย'
        }
        feat_df['Feature_TH'] = feat_df['Feature'].map(feature_names_thai)

        # สร้าง Tab สำหรับ visualization แบบต่างๆ
        tab1, tab2, tab3 = st.tabs(["📊 กราฟแท่ง", "🎯 แบบโดนัท", "📈 แบบ Radar"])
        
        with tab1:
            # Interactive Bar Chart ด้วย Plotly
            fig_bar = px.bar(
                feat_df, 
                x='Importance', 
                y='Feature_TH',
                orientation='h',
                title='📊 ปัจจัยสำคัญต่อการทำนายของ AI (แบบ Interactive)',
                labels={'Importance': 'ระดับความสำคัญ', 'Feature_TH': 'ปัจจัย'},
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, width='stretch')
            
            # เพิ่มข้อมูลเชิงลึก
            st.info("📌 **คลิกที่แท่งกราฟ** เพื่อดูรายละเอียดของแต่ละปัจจัย")
            
        with tab2:
            # Donut Chart
            fig_donut = go.Figure(data=[go.Pie(
                labels=feat_df['Feature_TH'],
                values=feat_df['Importance'],
                hole=0.4,
                marker_colors=px.colors.sequential.Viridis
            )])
            fig_donut.update_layout(
                title='🎯 สัดส่วนความสำคัญของปัจจัยต่างๆ',
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_donut, width='stretch')
            
        with tab3:
            # Radar Chart
            fig_radar = go.Figure()
            
            # Normalized values สำหรับ Radar Chart
            max_importance = feat_df['Importance'].max()
            normalized_importance = (feat_df['Importance'] / max_importance) * 100
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_importance,
                theta=feat_df['Feature_TH'],
                fill='toself',
                name='ความสำคัญ',
                line_color='rgb(67, 67, 67)',
                fillcolor='rgba(67, 67, 67, 0.25)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title='📈 ภาพรวมปัจจัยสำคัญ (Radar Chart)',
                height=500
            )
            st.plotly_chart(fig_radar, width='stretch')
            
        # เพิ่มส่วนวิเคราะห์เชิงลึก
        st.subheader("🔍 วิเคราะห์เชิงลึก")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🏆 ปัจจัยสำคัญที่สุด", feat_df.iloc[0]['Feature_TH'], f"{feat_df.iloc[0]['Importance']:.3f}")
            st.metric("📉 ปัจจัยที่น้อยที่สุด", feat_df.iloc[-1]['Feature_TH'], f"{feat_df.iloc[-1]['Importance']:.3f}")
            
        with col2:
            # คำนวณสัดส่วน
            top_3_sum = feat_df.head(3)['Importance'].sum()
            total_sum = feat_df['Importance'].sum()
            top_3_percentage = (top_3_sum / total_sum) * 100
            
            st.metric("📊 3 ปัจจัยแรกรวม", f"{top_3_percentage:.1f}%", "ของความสำคัญทั้งหมด")
            st.metric("🔢 จำนวนปัจจัย", len(feat_df), "ตัว")
            
        # แสดงตารางรายละเอียด
        st.markdown("### 📋 ตารางรายละเอียดความสำคัญ")
        display_df = feat_df[['Feature_TH', 'Importance']].copy()
        display_df['Importance_%'] = (display_df['Importance'] / display_df['Importance'].sum() * 100).round(2)
        display_df.columns = ['ปัจจัย', 'ค่าความสำคัญ', 'สัดส่วน (%)']
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        # ------------------------------------------
        # ส่วนที่ 4: ดาวน์โหลดข้อมูลสรุปผล (Export Section)
        # ------------------------------------------
        st.divider()
        st.subheader("📥 ดาวน์โหลดข้อมูลสรุปผล")
        
        # สร้างข้อมูลสรุปสำหรับดาวน์โหลด
        export_data = input_data.copy()
        export_data.columns = ['Daily_Return_%', 'SMA_50_USD', 'SMA_200_USD', 'RSI_14', 'MACD', 'Volatility', 'Volume_BTC']
        export_data['Daily_Return_%'] = export_data['Daily_Return_%'] * 100  # แปลงกลับเป็น %
        
        # เพิ่มข้อมูลผลการทำนาย
        if prediction is not None:
            export_data['Prediction'] = 'ขาขึ้น' if prediction == 1 else 'ขาลง'
            export_data['Confidence_%'] = prob_up if prediction == 1 else prob_down
        
        csv = export_data.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลสรุปผล (CSV)",
            data=csv,
            file_name='btc_prediction_report.csv',
            mime='text/csv',
            help="ดาวน์โหลดข้อมูลที่ใช้ในการทำนายและผลลัพธ์"
        )
        
    except Exception as e:
        st.warning("โมเดลไม่รองรับการแสดง Feature Importance หรือโครงสร้าง Pipeline ไม่ตรงกัน")

else:
    st.warning("รอการโหลดโมเดล หรือกรุณาแก้ไขตัวเลขในช่องกรอกข้อมูลด้านบนให้ถูกต้องครับ...")