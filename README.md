# 📈 Bitcoin Price Trend Predictor (AI Trading Assistant)

โปรเจค Machine Learning สำหรับทำนายทิศทางราคาบิตคอยน์ (Bitcoin) ล่วงหน้า 1 วัน โดยใช้ข้อมูลในอดีตมาสร้าง Technical Indicators เพื่อวิเคราะห์โมเมนตัมและแนวโน้มของตลาด 

โปรเจคนี้เป็นส่วนหนึ่งของวิชา ML Deployment Project ที่ครอบคลุมตั้งแต่กระบวนการ EDA, การทำ Feature Engineering, การเปรียบเทียบโมเดล และการ Deploy ออกมาเป็น Web Application

## 🚀 ฟีเจอร์หลัก (Key Features)
- **Advanced EDA:** วิเคราะห์พฤติกรรมของราคา, ความผันผวน (Volatility) และความสัมพันธ์ของอินดิเคเตอร์
- **Time-Series Validation:** ใช้ `TimeSeriesSplit` ในการแบ่งข้อมูลทดสอบเพื่อป้องกัน Data Leakage
- **Model Comparison:** แข่งขันทรรศนะระหว่าง **XGBoost (Boosting)** และ **Random Forest (Bagging)** - **Business Metric:** โฟกัสการประเมินผลด้วยค่า **Precision** เพื่อลดความเสี่ยงจากสัญญาณซื้อหลอก (False Positive)
- **Web Deployment:** นำโมเดล XGBoost ที่ชนะเลิศมา Deploy ผ่าน **Streamlit** พร้อมแสดงค่าความน่าจะเป็น (Probability) และ Feature Importance

## 📁 โครงสร้างโปรเจค (Project Structure)
- `BTC_Project.ipynb`: ไฟล์ Source Code หลักที่ใช้ในการทำ Data Prep, EDA, จูนพารามิเตอร์ และประเมินผล
- `app.py`: โค้ดสำหรับสร้างหน้า Web Application ด้วย Streamlit
- `best_bitcoin_model.pkl`: โมเดล XGBoost ที่ผ่านการเทรนและปรับจูนพารามิเตอร์แล้ว
- `requirements.txt`: รายการไลบรารีทั้งหมดที่จำเป็นต่อการรันโปรเจค

## 💻 วิธีการติดตั้งและใช้งาน (How to Run)
1. Clone Repository นี้ลงเครื่องของคุณ:
   ```bash
   git clone [https://github.com/ช](https://github.com/ช)ื่อยูสเซอร์ของคุณ/ชื่อโปรเจค.git
   ติดตั้งไลบรารีที่จำเป็น:

Bash
pip install -r requirements.txt
รันแอปพลิเคชัน Streamlit:

Bash
streamlit run app.py
