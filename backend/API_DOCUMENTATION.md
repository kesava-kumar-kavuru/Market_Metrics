# API Documentation

This document describes the API endpoints available in the Quantum ML Backend.

## Base URL

By default, the backend runs on:
`http://localhost:5000`

## Endpoints

### 1. Home Check
**URL:** `/`
**Method:** `GET`
**Description:** A simple health check route to verify if the backend server is running.

**Response:**
- **Status Code:** 200 OK
- **Body:** `Quantum ML Backend is running!`

---

### 2. Get Predictions
**URL:** `/api/predictions`
**Method:** `GET`
**Description:** Fetches prediction data for the last 30 days of the dataset. It runs the pre-trained SVM and VQC (Variational Quantum Classifier) models on the historical data to generate buy/sell signals and price predictions.

**Response:**
- **Status Code:** 200 OK
- **Content-Type:** `application/json`
- **Body:** An array of prediction objects.

**Response Structure:**

The response is a JSON array where each object represents a daily prediction.

```json
[
  {
    "date": "2023-10-27",
    "actual": 7291.28,
    "vqc_prediction": 7400.65,
    "svm_prediction": 7400.65,
    "confidence": 0.92,
    "signal": "BUY"
  },
  ...
  {
    "date": "2023-12-08",
    "actual": 7554.47,
    "vqc_prediction": 7441.15,
    "svm_prediction": 7441.15,
    "confidence": 0.89,
    "signal": "SELL",
    "accuracy": 86.7  // Only present in the last object
  }
]
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `date` | String | The date of the record (YYYY-MM-DD). |
| `actual` | Float | The actual closing price of the asset. |
| `vqc_prediction` | Float | The predicted price based on the VQC model's classification. |
| `svm_prediction` | Float | The predicted price based on the SVM model's classification. |
| `confidence` | Float | The confidence score of the prediction (0.0 to 1.0). |
| `signal` | String | The trading signal: "BUY" or "SELL". |
| `accuracy` | Float | (Last item only) The classification accuracy percentage over the returned period. |

**Error Response:**

If the data file is missing or cannot be processed:

```json
{
  "error": "Data file not found at data/dataset.csv"
}
```
