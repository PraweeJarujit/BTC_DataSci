# 📈 Bitcoin AI Trading Assistant

แอปพลิเคชันปัญญาประดิษฐ์ (AI) สำหรับทำนายทิศทางราคาบิตคอยน์ล่วงหน้า 1 วัน ด้วย Machine Learning (XGBoost) พร้อม visualization ขั้นสูงและ UI ที่ใช้งานง่าย

## 🌟 Live Demo
**🚀 ลองใช้งานได้ที่:** [https://btc-data-science.streamlit.app/](https://btc-data-science.streamlit.app/)

## ✨ ฟีเจอร์เด่น (Advanced Features)

### 🤖 การทำนายอัจฉริยะ
- **AI Prediction Engine**: ใช้ XGBoost วิเคราะห์ Technical Indicators 7 ตัว
- **Confidence Level**: แสดงความมั่นใจของ AI เป็นเปอร์เซ็นต์
- **Risk Assessment**: ระบุระดับความเสี่ยง (ต่ำ/ปานกลาง/สูง)
- **Smart Suggestions**: แนะนำกลยุทธ์การลงทุนตามสถานการณ์

### 📊 Visualization ขั้นสูง
- **📈 Interactive Bar Chart**: กราฟแท่งแบบ interactive ด้วย Plotly
- **🎯 Donut Chart**: แสดงสัดส่วนความสำคัญของปัจจัยต่างๆ
- **🕸️ Radar Chart**: ภาพรวมปัจจัยสำคัญแบบ 360 องศา
- **🎚️ Gauge Chart**: แสดงระดับความมั่นใจแบบดิจิทัล
- **📋 Market Analysis**: วิเคราะห์สภาพตลาดแบบ real-time

### 🛡️ ความปลอดภัย
- **Input Validation**: ตรวจสอบค่าที่กรอกไม่ให้ผิดปกติ
- **Disclaimer ครบถ้วน**: เตือนความเสี่ยงอย่างชัดเจน
- **Educational UI**: อธิบาย Technical Indicators เป็นภาษาที่เข้าใจง่าย

## 🎯 Technical Indicators ที่ใช้

| Indicator | คำอธิบาย | ช่วงค่า |
|-----------|------------|----------|
| 📊 Daily Return | การเปลี่ยนแปลงราคาในวันนี้ | -20% ถึง +20% |
| 📈 SMA 50/200 | ราคาเฉลี่ยเคลื่อนที่ 50/200 วัน | 1,000-200,000 USD |
| 🔄 RSI 14 | ดัชนีแรงซื้อแรงขาย | 0-100 |
| ⚡ MACD | ความเร็วการเปลี่ยนแปลงราคา | ±5,000 |
| 📊 Volatility | ความผันผวนของราคา | 0-0.20 |
| 💰 Volume | ปริมาณการซื้อขาย | 0-1,000,000 BTC |

## 🚀 วิธีการติดตั้งและใช้งาน

### 1. Clone Repository
```bash
git clone https://github.com/PraweeJarujit/BTC_DataSci.git
cd BTC_DataSci
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. รันแอปพลิเคชัน
```bash
streamlit run app.py
```

### 4. เปิดเบราว์เซอร์
- Local: http://localhost:8501
- Network: http://192.168.x.x:8501

## 📁 โครงสร้างโปรเจค

```
BTC_DataSci/
├── app.py                    # Streamlit Web Application
├── best_bitcoin_model.pkl    # XGBoost Model (Trained)
├── BTC_Project.ipynb         # Jupyter Notebook (EDA & Training)
├── requirements.txt           # Python Dependencies
├── packages.txt              # Streamlit Cloud Dependencies
├── README.md                 # Project Documentation
└── .streamlit/
    └── config.toml           # Streamlit Configuration
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy

### Frontend
- **Streamlit**: Web Framework
- **Plotly**: Interactive Charts
- **Matplotlib/Seaborn**: Static Visualizations

### Deployment
- **Streamlit Community Cloud**
- **GitHub Integration**

## 📊 Model Performance

- **Algorithm**: XGBoost (Gradient Boosting)
- **Validation**: TimeSeriesSplit (Prevent Data Leakage)
- **Focus Metric**: Precision (Reduce False Positives)
- **Features**: 7 Technical Indicators
- **Accuracy**: ~85% (Historical Performance)

## 🎮 วิธีใช้งาน

### ขั้นตอนที่ 1: กรอกข้อมูลตลาด
1. ใช้ Sidebar ด้านข้างขวา
2. กรอกค่า Technical Indicators ปัจจุบัน
3. ระบบจะตรวจสอบค่าอัตโนมัติ

### ขั้นตอนที่ 2: ดูผลการทำนาย
1. AI จะวิเคราะห์ข้อมูลทันที
2. แสดงทิศทางราคา (ขึ้น/ลง)
3. โชว์ความมั่นใจเป็นเปอร์เซ็นต์
4. แนะนำกลยุทธ์การลงทุน

### ขั้นตอนที่ 3: วิเคราะห์เชิงลึก
1. ดู Feature Importance แบบ interactive
2. ตรวจสอบ Market Condition Summary
3. ประเมินระดับความเสี่ยง

## ⚠️ ข้อควรระวังสำคัญ (Disclaimer)

### ❌ สิ่งที่แอปนี้ไม่ใช่
- ❌ การการันตีว่าจะได้กำไร 100%
- ❌ คำแนะนำการลงทุนที่ต้องตามอย่างเคร่งครัด
- ❌ เครื่องมือทำนายอนาคตที่แม่นยำเสมอไป

### ✅ สิ่งที่แอปนี้คือ
- ✅ เครื่องมือช่วยวิเคราะห์ข้อมูลตลาด
- ✅ ข้อมูลเพิ่มเติมสำหรับประกอบการตัดสินใจ
- ✅ แนวทางในการบริหารความเสี่ยง

### 📊 ความน่าจะเป็น (Probability) คืออะไร?
- ค่าเปอร์เซ็นต์ที่แสดงคือ **ความมั่นใจของ AI** อิงจากสถิติในอดีต
- เช่น 85% หมายถึง AI มั่นใจ 85% ว่าจะเป็นไปตามที่ทาย
- **ไม่ใช่** การการันตีว่าจะถูกต้อง 85%

### 🎯 หลักการใช้งานอย่างปลอดภัย
1. **ใช้เป็นข้อมูลเสริม** อย่าพึ่งพาอย่างเดียว
2. **ตั้ง Stop-loss** ทุกครั้งที่เทรด
3. **ลงทุนเท่าที่สามารถขาดทุนได้**
4. **ศึกษาข้อมูลจากแหล่งอื่น** ประกอบด้วย
5. **ไม่ลงทุนเมื่อไม่เข้าใจ** ความเสี่ยงที่เผชิญ

## 🤝 ผู้มีส่วนร่วม

- **Developer**: Prawee Jarujit
- **Project**: Machine Learning Deployment
- **Framework**: Streamlit + XGBoost

## 📄 License

This project is for educational purposes only. Please use responsibly and at your own risk.

---

**⚠️ การลงทุนมีความเสี่ยง ผู้ลงทุนควรศึกษาข้อมูลก่อนตัดสินใจ**

**🌟 ถูกใจฟีเจอร์นี้? ให้ ⭐ บน GitHub หน่อย!**
