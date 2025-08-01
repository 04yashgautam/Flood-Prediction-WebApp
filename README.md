# 🌊 Flood Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web--App-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Flood-Prediction-WebApp** is an interactive Python application that predicts flood risk based on real-time user inputs of hydrological and meteorological parameters. It combines a trained **MobileNet-based deep learning model** with a simple **CLI** and **Flask web interface** for instant, on-the-fly predictions.

---

## 📋 Table of Contents

1. [🚀 Quick Start](#-quick-start)
2. [🔍 Project Highlights](#-project-highlights)
3. [⚙️ Configuration](#️-configuration)
4. [📦 requirements.txt](#-requirementstxt)
5. [📫 Contact](#-contact)

---

## 🚀 Quick Start

1. **Clone** the repo

   ```bash
   git clone https://github.com/04yashgautam/flood-predictor.git
   cd flood-predictor
   ```
2. **Setup** environment

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. **Run** the application

   ```bash
   python app.py
   ```
4. **Input** parameters (e.g., rainfall, river level) when prompted and receive a flood risk score.

---

## 🔍 Project Highlights

* 🧠 **Image Classification Support**: Integrates **MobileNet** pre-trained on **ImageNet** for optional image-based flood classification or visual data interpretation.

* ♻️ **Modular & Flexible**: Easily extendable with new features or model updates.

---

### 🖥️ CLI Mode

```bash
$ python app.py
Enter rainfall (mm): 25
Enter river level (m): 3.2
Enter soil moisture (%): 45
🚨 Flood Risk Level: **High** (0.82)
```

### 🌐 Web Mode

```bash
$ python app.py --web
 * Running on http://127.0.0.1:5000/
```

Open your browser and fill out the form to view the results.

---

## 📈 How It Works

1. **User Input** → Inputs (rainfall, river level, etc.) are validated and scaled.
2. **Prediction** → Model returns risk score between 0 (Low) and 1 (High).
3. **Output** → Categorized into risk levels: Low, Moderate, High.

---

## 📦 requirements.txt

```txt
tensorflow>=2.5.0
keras>=2.4.0
flask>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

---

## 📫 Contact

For questions, feedback, or collaboration ideas, reach out via [GitHub](https://github.com/04yashgautam).

---
