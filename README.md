# Customer Sentiment Analysis using NLP

HEAD
## Project Overview
This project aims to analyze customer reviews using Natural Language Processing (NLP) techniques.
The main objectives are:
- Sentiment classification (positive / negative)
- Identification of key aspects influencing customer satisfaction
- Model interpretability and insights extraction

## Technologies Used
- Python
- Scikit-learn
- NLP (TF-IDF)
- Streamlit (for demo)
- Matplotlib / Seaborn

## Project Status
🚧 In progress – Academic project (Data Science)

## Author
Mouad Souhal

This ZIP is **ready to run** on Windows with VS Code.

## 1) Setup (PowerShell)
From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Run the notebook
Open: `notebooks/01_eda_and_baseline.ipynb`
- In VS Code: Click **Select Kernel** and choose `.venv`

## 3) Run from command line
```powershell
python src/main.py
```

## 4) Run Streamlit demo (optional)
```powershell
streamlit run app/app.py
```

Dataset is included at: `data/Amazon_Reviews.csv`
4c0ba92 (Initial commit: project structure + notebook + app)
