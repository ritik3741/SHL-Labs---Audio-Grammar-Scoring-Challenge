# 🎙️ SHL Audio Grammar Scoring Challenge

End-to-End Machine Learning Pipeline for Automated Spoken Grammar Assessment

---

## 📌 Project Overview

This project provides a complete solution for the [SHL Labs – Audio Grammar Scoring Challenge](https://www.kaggle.com/competitions/shl-audio-scoring-challenge/overview). The goal is to predict a continuous grammar score (0–5) from spoken English audio samples (45–60 seconds).

The solution combines:

* 🎧 Acoustic & prosodic features
* 🧠 Deep speech embeddings (Wav2Vec2)
* 📝 Linguistic features from Whisper transcription
* 🤖 Ensemble machine learning models
* 📊 Calibration & evaluation dashboard

The main implementation notebook is available in the repository:
`deepseek_python_20260303_5f71ab.py`
Referenced file: fileciteturn0file0

---

## 📊 Dataset Information

* **Training Samples:** 409 audio files
* **Test Samples:** 197 audio files
* **Score Range:** 0–5 (Likert scale)
* **Evaluation Metrics:**

  * Pearson Correlation
  * RMSE (Root Mean Squared Error)

---

## 🏗️ Solution Architecture

The pipeline follows a structured end-to-end workflow:

### 1️⃣ Data Loading

* Reads original `train.csv` and `test.csv`
* Preserves exact filename formatting for submission

---

### 2️⃣ Feature Engineering

#### 🎧 Acoustic Features (Librosa)

* MFCCs (13 + delta)
* Spectral centroid, bandwidth, rolloff
* Zero-crossing rate
* RMS energy statistics
* Pitch features using `pyin`
* Voiced ratio & silence ratio

These features capture pronunciation quality, articulation, fluency, and speech continuity.

---

#### 🧠 Deep Speech Embeddings (Wav2Vec2)

* Pretrained `facebook/wav2vec2-base-960h`
* Mean pooled last hidden state
* First 50 dimensions retained (to reduce overfitting)

These embeddings encode high-level acoustic patterns.

---

#### 📝 Linguistic Features (Whisper + spaCy + LanguageTool)

1. Whisper → Speech-to-text transcription
2. LanguageTool → Grammar error detection
3. spaCy → POS tagging & lexical diversity

Extracted features include:

* Word count
* Sentence count
* Grammar error count & rate
* POS ratios (noun/verb/adj)
* Type-token ratio
* Average word length

---

### 3️⃣ Feature Selection

* Remove constant features
* Fill missing values
* Correlation-based filtering (|r| > 0.05)
* Standard scaling

---

### 4️⃣ Model Ensemble

Five diverse models are used:

* XGBoost
* LightGBM
* Gradient Boosting Regressor
* Random Forest
* Ridge Regression

#### 🔹 Strategy

* 5-Fold Cross Validation
* Inverse-RMSE weighted ensemble
* Isotonic Regression calibration

This ensures better generalization and improved score distribution alignment.

---

## 📈 Evaluation

### Cross-Validation Metrics

* Individual model RMSE & Pearson
* Ensemble RMSE & Pearson

### Required Metric

* Training RMSE (explicitly computed and printed)

### Visualizations Generated

* Actual vs Predicted scatter plot
* Residual distribution
* Model comparison chart
* Calibration curve
* Feature importance plot
* Error analysis (hardest samples)

---

## 📤 Submission Generation

Critical requirement handled carefully:

* Original filename format from `test.csv` is preserved
* Predictions clipped to [0, 5]
* Rounded to 4 decimal places
* Saved as `submission.csv`

Verification steps included:

* Filename match check
* Prediction count validation
* Score range confirmation

---

## 🧠 Key Insights

1. Deep speech embeddings significantly improve performance.
2. Linguistic error rate strongly correlates with grammar score.
3. Ensemble outperforms individual models.
4. Calibration improves performance at score extremes.

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install openai-whisper language-tool-python librosa soundfile xgboost transformers torch lightgbm spacy
python -m spacy download en_core_web_sm
```

2. Set dataset paths correctly.
3. Run the notebook/script from top to bottom.
4. Final `submission.csv` will be generated in working directory.

---

## 📂 Output Files Generated

* `train_features.csv`
* `test_features.csv`
* `score_distribution.png`
* `feature_correlations.png`
* `evaluation_dashboard.png`
* `submission.csv`

---

## 🎯 Conclusion

This solution demonstrates that combining:

* Acoustic features
* Transformer-based speech embeddings
* Linguistic grammar analysis
* Regularized ensemble modeling

results in a strong and robust grammar scoring engine.

The pipeline is modular, interpretable, and competition-ready.

---

If you want, I can also create:

* 🔹 A GitHub-optimized version of this README
* 🔹 A shorter Kaggle competition description version
* 🔹 A research-style technical documentation version
