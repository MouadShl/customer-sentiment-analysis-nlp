# 💬 Customer Review Sentiment Analysis (NLP)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20LogReg-success)
![License](https://img.shields.io/badge/License-MIT-green)

**Analyse de sentiment de commentaires clients** • **NLP & Machine Learning** • **Démo Streamlit en ligne**

Détection automatique **positif / négatif** • Interprétation des mots importants • Pipeline complet prêt production

🌐 **Application Live** • 📖 **Documentation** • 🚀 **Démo Express** • 💡 **Insights**

---

## 🎯 Executive Summary
Une solution **end-to-end** de Data Science (NLP) pour analyser des avis clients et prédire le sentiment, avec :
- 🔍 **Exploration & nettoyage** (EDA + preprocessing)
- 🧠 **Modèle ML baseline solide** (TF-IDF + Logistic Regression)
- 📊 **Interprétabilité** (mots influents + insights)
- 🌐 **Démo interactive** (Streamlit) accessible à tous

---

## 🌐 Application Live — Testez Maintenant !
🚀 **DÉMO LIVE :** `YOUR_STREAMLIT_APP_URL`

✅ Zero configuration • Interface simple • Résultat instantané

### ⚡ Test Express (30 secondes)
1. Ouvrez le lien **Live**
2. Collez un commentaire (en anglais ou texte simple)
3. Cliquez **Predict**
4. Obtenez le sentiment **POSITIVE / NEGATIVE**

---

## 💡 Innovation Différenciante
### 🎯 Ce qui rend ce projet “pro”
- **Pipeline propre** (reproductible + structuré)
- **Interprétabilité** : extraction des mots qui poussent la décision
- **Déploiement cloud** : démonstration live (sans installer le code)
- **Dataset externe** téléchargé automatiquement (GitHub Release) — repo léger & clean

---

## 🏗 Stack Technique
- **Frontend** : Streamlit (UI)
- **Backend** : Python
- **NLP** : TF-IDF (uni/bi-gram)
- **ML** : Logistic Regression (class_weight balanced)
- **Data** : Pandas / NumPy
- **Déploiement** : Streamlit Cloud
- **Dataset** : GitHub Release asset (auto-download)

---

## 🧠 Modèle & Méthodologie
### Pipeline
1) Text preprocessing (clean, drop NA)  
2) Feature extraction: **TF-IDF** (max_features=5000, ngram_range=(1,2))  
3) Classification: **Logistic Regression**  
4) Output: **positive / negative**  
5) Interpretation: **top weighted words** (insights)

---

## 📁 Structure du Projet
```text
customer-sentiment-analysis-nlp/
├── app/
│   └── app.py                 # Streamlit web app (live demo)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
├── src/
│   └── main.py                # CLI run (optional)
├── data/                      # local only (ignored in git)
├── requirements.txt
└── README.md
