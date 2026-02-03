# Network Throughput Forecasting with LSTM

This project implements a Deep Learning model specifically designed to forecast network traffic volume (throughput) using Long Short-Term Memory (LSTM) networks. The primary goal is to predict bandwidth usage for the next second based on historical traffic patterns.

---

## 1. Project Overview

Predicting network throughput is critical for resource allocation and congestion management. This implementation transitions from simple packet classification to advanced time-series forecasting.

### 1.1 Data Source
The model uses network traffic capture data (CSV format) containing packets with timestamps and lengths.

---

## 2. Data Preprocessing Pipeline

Effective forecasting in network traffic requires specialized preprocessing due to the highly bursty and non-linear nature of the data.

### 2.1 Time-Series Aggregation
Raw packet data is aggregated into 1-second intervals by summing the `Length` of all packets within each second. This transforms discrete events into a continuous time-series signal.

### 2.2 Feature Engineering and Scaling
*   **Log Transformation (np.log1p):** Network traffic often contains extreme spikes (outliers). Applying a natural log transformation compresses these spikes, preventing them from dominating the loss function and allowing the model to learn from smaller, more frequent patterns.
*   **Data Smoothing (Rolling Mean):** A 3-period rolling average is applied to reduce high-frequency noise (jitter), emphasizing the underlying trend.
*   **Sliding Window:** We use a `WINDOW_SIZE` of 30 seconds. The model looks at the past 30 data points to predict the very next value.

---

## 3. Model Architecture

We utilize a **Stacked Bidirectional LSTM** architecture to capture long-term dependencies in the traffic data.

### 3.1 Neural Network Layers
*   **Bidirectional LSTM (64 units):** Unlike standard LSTMs, the bidirectional layer processes the 30-second sequence in both forward and backward directions, capturing more context from the sequence.
*   **Dropout (0.2):** Prevents overfitting by randomly setting 20% of input units to 0 during training.
*   **LSTM (32 units):** A second LSTM layer to further refine the sequence representation.
*   **Dense (16 units, ReLU):** A fully connected layer that extracts higher-level features from the LSTM outputs.
*   **Dense (1 unit):** The final output layer providing the predicted throughput value for the next second.

---

## 4. Training and Evaluation

### 4.1 Loss Function
The model is compiled using **Mean Absolute Error (MAE)** instead of the standard Mean Squared Error (MSE).
*   **Rationale:** MSE penalizes large errors exponentially (due to squaring), which makes the model overly sensitive to traffic spikes. MAE is more robust to these spikes, resulting in a model that follows the actual trend more closely.

### 4.2 Metrics
Predictions are transformed back from the log scale (using `np.expm1`) to their original Byte units for real-world interpretation and benchmarking against ground truth data.

---

## 5. Usage for Network Maintenance

The output of the model can be used for **Predictive Alerting**:
*   If the forecasted throughput exceeds a pre-defined threshold (e.g., 3x the average), the system can trigger an automated alert to prevent potential network congestion before it occurs.
