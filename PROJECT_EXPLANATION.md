# Network Throughput Forecasting: Technical Documentation

เอกสารฉบับนี้อธิบายรายละเอียดการทำงานของโครงสร้างโค้ดและการออกแบบโมเดล Deep Learning ในไฟล์ `main_pipeline_2.ipynb` สำหรับการทำนายปริมาณข้อมูลในเครือข่ายล่วงหน้า

---

## 1. Data Preparation (การเตรียมข้อมูล)

โครงการนี้ใช้ข้อมูลจาก [Network Traffic Dataset](https://www.kaggle.com/datasets/ravikumargattu/network-traffic-dataset) บน Kaggle โดยไฟล์หลักที่ใช้ในการวิเคราะห์และเทรนโมเดลคือ `Midterm_53_group.csv`

ขั้นตอนนี้สำคัญที่สุด เพราะ "Garbage In, Garbage Out" หากข้อมูลไม่ดี โมเดลก็จะทายผิด


### 1.1 Time-Series Aggregation
```python
df['Time_Sec'] = df['Time'].astype(int)
series = df.groupby('Time_Sec')['Length'].sum().values
```
*   **อธิบาย:** ข้อมูลดิบ (Raw Data) มาเป็นราย Packet ซึ่งไม่ต่อเนื่อง เราต้องรวม (`sum`) ขนาดของทุก Packet (`Length`) ที่อยู่ในวินาทีเดียวกัน (`Time_Sec`) เพื่อสร้างเส้นกราฟ **Throughput (Bytes per Second)**

### 1.2 Mathematical Tricks (เทคนิคการจัดการข้อมูลกระชาก)
```python
log_series = np.log1p(series)
log_series = pd.Series(log_series).rolling(window=3).mean().dropna().values
```
*   **np.log1p (Log Transformation):** ข้อมูล Network ชอบมี Spike (ยอดแหลมที่พุ่งสูงมาก) ซึ่งทำให้โมเดลเรียนรู้ยาก การทำ Log จะช่วยบีบจากช่วง 0 - 3,000,000 ให้เหลือประมาณ 0 - 15 ทำให้โมเดล "จ่ายค่าน้ำหนัก" ให้จุดเล็กๆ ได้ดีขึ้น
*   **Rolling Mean (Smoothing):** เราใช้ค่าเฉลี่ยเคลื่อนที่ 3 วินาที เพื่อลดสัญญาณรบกวน (Jitter) ทำให้โมเดลเห็น "แนวโน้ม" (Trend) มากกว่าความถี่ยิบย่อย

### 1.3 Windowing (การสร้างหน้าต่างข้อมูล)
```python
WINDOW_SIZE = 30
# ... ในฟังก์ชัน create_windows ...
X.append(data[i:i+window_size]) # Input: 30 วินาทีก่อนหน้า
y.append(data[i+window_size])    # Target: วินาทีถัดไป
```
*   **อธิบาย:** เราใช้แนวคิด **Sliding Window** โดยให้โมเดลดูอดีต 30 วินาที เพื่อทายผลใน 1 วินาทีข้างหน้า
*   **Data Shape:** ต้องปรับข้อมูลเป็น 3 มิติ `(samples, 30, 1)` เพราะชั้น LSTM ต้องการรับข้อมูลแบบ [จำนวนชุด, ลำดับเวลา, จำนวนฟีเจอร์]

---

## 2. Model Architecture Design (การออกแบบโมเดล)

เราใช้สถาปัตยกรรมแบบ **Stacked Recurrent Neural Network (RNN)**

```python
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
```

### รายละเอียดแต่ละ Layer:
1.  **Bidirectional LSTM (64 units):** 
    *   *ทำไมต้องใช้?* LSTM ปกติจะมองไปข้างหน้าอย่างเดียว แต่ Bidirectional จะช่วยให้โมเดลประมวลผลลำดับข้อมูลทั้ง "เดินหน้า" และ "ถอยหลัง" ภายในหน้าต่าง 30 วิ ทำให้เห็นความสัมพันธ์ของ Pattern ได้ลึกซึ้งกว่า
    *   *return_sequences=True:* เพื่อส่งต่อข้อมูลทั้งลำดับเวลาไปยัง Layer ถัดไป
2.  **Dropout (0.2):**
    *   *ทำไมต้องใช้?* สุ่ม "ปิด" Neural Network 20% ระหว่างเทรน เพื่อป้องกัน **Overfitting** (ป้องกันไม่ให้โมเดลจำข้อสอบ แต่เน้นให้โมเดลเข้าใจหลักการ)
3.  **LSTM (32 units):**
    *   ช่วยสรุปผลจากชั้นแรกให้เหลือใจความสำคัญก่อนส่งต่อ
4.  **Dense (16 units + ReLU):**
    *   เป็น Fully Connected Layer เพื่อช่วยในการตัดสินใจขั้นสุดท้าย โดยใช้ **ReLU** เพื่อตัดค่าติดลบและเพิ่มความเป็น Non-linear
5.  **Dense (1 unit):**
    *   Output ตัวเดียวคือค่า Throughput (Bytes) ที่เราต้องการทำนาย

---

## 3. Loss Function & Optimizer

```python
model.compile(optimizer='adam', loss='mae')
```
*   **Optimizer (Adam):** เป็นมาตรฐานที่เทพที่สุดในปัจจุบัน เพราะปรับ Learing Rate ให้อัตโนมัติ เทรนไวและเสถียร
*   **Loss Function (MAE):** เราเลือกใช้ **Mean Absolute Error** แทน MSE (มาตรฐานทั่วไป) 
    *   *เหตุผล:* MSE จะยกกำลังสองค่าที่ทายผิด ทำให้จุดที่ผิดเยอะๆ (Spikes) มีผลรุนแรงมากเกินไป จนโมเดลกลัวและไม่ยอมขยับตามกราฟ (เกิดอาการเส้นตรงแบบอันแรก) ส่วน **MAE** จะใจดีกว่า ทำให้โมเดลกล้าเลื้อยตามเส้นกราฟจริงได้ดีกว่า

---

## 4. Evaluation (การประเมินผล)

```python
pred_inv = np.expm1(predictions)
true_inv = np.expm1(y_test)
```
*   **Inverse Transformation:** เนื่องจากตอนแรกเราทำ Log เราจึงต้องทำ "ส่วนกลับ" ด้วย `expm1` (Exponential - 1) เพื่อให้ได้ค่าออกมาเป็นหน่วย **Bytes** ที่มนุษย์อ่านรู้เรื่อง

---

## 5. สรุปความแตกต่างของเวอร์ชัน (RATIONALE)
*   **Baseline Pipeline (main_pipeline.ipynb):** ใช้ MSE + ข้อมูลดิบ -> ผลคือโมเดลทายแต่ค่าเฉลี่ย เพราะกลัวค่า Spike
*   **Pipeline 2 (main_pipeline_2.ipynb):** ใช้ Log + Smoothing + MAE + Bidirectional -> ผลคือโมเดลเห็น Trend ของเน็ตชัดขึ้นและกล้าทายค่าตามการขึ้นลงจริงของข้อมูล
