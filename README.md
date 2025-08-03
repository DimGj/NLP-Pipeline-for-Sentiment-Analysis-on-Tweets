# 🐦 NLP Pipeline for Sentiment Analysis on Tweets

This project implements a complete **Natural Language Processing (NLP) pipeline** to perform **sentiment analysis** on tweets.  
It was developed as part of the **Data Extraction Methods** university course, in collaboration with **[@Natedrake7](https://github.com/Natedrake7)**.

---

## 📌 Features
- 🧹 **Data preprocessing & cleaning** (tokenization, stopword removal, normalization).  
- 🔠 **Vectorization of tweets** using techniques like **TF-IDF** and **Bag-of-Words**.  
- 🤖 **Classification models** for sentiment detection (positive, negative, neutral).  
- 📊 **Exploratory Data Analysis (EDA)** with visual insights.  
- 📝 **Jupyter Notebook (`final.ipynb`)** for experiments and demonstrations.  
- 🚀 **Main script (`main.py`)** to run the full pipeline.  

---

## 📂 Project Structure
```

NLP-Pipeline-for-Sentiment-Analysis-on-Tweets/
│── README.md                 # Project documentation
│── CodeScripts/              # Core codebase
│   ├── main.py               # Entry point for pipeline execution
│   ├── imports.py            # Common imports & constants
│   ├── FileHandlers.py       # File input/output utilities
│   ├── DataAnalysis.py       # Exploratory Data Analysis scripts
│   ├── VectorizationTweets.py# Vectorization (TF-IDF, BoW, etc.)
│   ├── Classification.py     # Sentiment classification models
│   └── final.ipynb           # Jupyter Notebook for experiments
│
│── PickleFiles/              # Serialized models/data
│   └── a.txt                 # (Placeholder/example pickle file with data)
│
│── TSVFiles/                 # Dataset files
│   └── a.txt                 # (Placeholder/example tsv file with data)
│

```

---

## ⚙️ Requirements

Make sure you have **Python 3.8+** installed.  
This project requires the following Python libraries:

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- nltk  
- jupyter (for running the notebook)  

---

## 🚀 Usage

- **Run the pipeline from the command line:**  
  ```bash
  python main.py
  ```

- **Open the Jupyter notebook for experiments:**

  ```bash
  jupyter notebook final.ipynb
  ```

---

## 🧩 Pipeline Workflow

1. **Load dataset** (tweets from TSV files).
2. **Preprocess data** (cleaning, tokenization, stopword removal).
3. **Vectorize tweets** into numerical features.
4. **Classify sentiments** using ML models.
5. **Generate results** with metrics and visualizations.

---

## 📊 Example Output

* Sentiment distribution of tweets (positive, negative, neutral).
* Evaluation metrics (accuracy, precision, recall, F1-score).
* Visualizations such as word frequencies or word clouds.



