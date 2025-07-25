# 🌊 Flood Prediction WebApp

**Flood-Prediction-WebApp** is an interactive Python application that predicts flood risk based on real‑time user inputs of hydrological and meteorological parameters. It blends deep learning models with a simple CLI/web interface for on‑the‑fly forecasting.

---

## 🚀 Quick Start

1. **Clone** the repo  
   ```bash
   git clone https://github.com/04yashgautam/Flood-Prediction-with-User-Inputs.git
   cd Flood-Prediction-with-User-Inputs
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
4. **Input** parameters (e.g., rainfall, river level) when prompted and receive flood risk score.

---

## 🔍 Project Highlights

- 🧠 **Deep Learning Core**: Utilizes an LSTM-based model trained on historical flood data.  
- 🖥️ **Interactive Interface**: CLI prompts or optional Flask web UI for data entry.  
- 📊 **Real‑Time Predictions**: Instant flood risk assessment on user-supplied inputs.  
- ♻️ **Modular Design**: Easily swap in new models or input features.

---

## ⚙️ Configuration

- **Thresholds**: Adjust risk thresholds in `src/utils.py`.  
- **Model Path**: Update model file location in `app.py` or via `--model` flag.  
- **Web Mode**: Launch Flask UI with `python app.py --web`.

---

## 🎛️ Usage Examples

### CLI Mode
```bash
$ python app.py
Enter rainfall (mm): 25
Enter river level (m): 3.2
Enter soil moisture (%): 45
🚨 Flood Risk Level: **High** (0.82)
```

### Web Mode
```
$ python app.py --web
 * Running on http://127.0.0.1:5000/
```
Open your browser and fill in the form to view results.

---

## 📈 How It Works

1. **User Input** → validated & scaled.  
2. **Feature Vector** → passed to trained LSTM model.  
3. **Prediction** → probability score between 0 (low) and 1 (high) risk.  
4. **Output** → risk category (Low, Moderate, High).

---
