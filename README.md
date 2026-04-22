# FRAG: Fair Retrieval-Augmented Generation for Sequential Recommendation

📝 Anonymous implementation for paper submission.

---

## 🚀 Overview
FRAG is a fairness-aware RAG framework for sequential recommendation.

✨ Key features:
- Adaptive retrieval with learnable threshold τ  
- Differentiable retriever via smooth relaxation  
- Online fairness-aware optimization  

🎯 Goal: balance recommendation utility and long-term group exposure.

---

## 🧠 Method
At each step:
1. Score items with retriever  
2. Select candidates via threshold τ  
3. Apply soft weights for differentiability  
4. Track group exposure over time  
5. Jointly optimize retriever and generator  

Training objective:
```
L = L_log + η L_reg
```

---

## 📂 Structure
```
data/        datasets  
models/      retriever & generator  
training/    training scripts  
evaluation/  metrics  
configs/     hyperparameters  
main.py      entry  
```

---

## ⚙️ Usage
Train:
```
python main.py --mode train --dataset movielens
```

Test:
```
python main.py --mode test --dataset movielens
```

---

## 📊 Datasets
- MovieLens  
- Steam  
- LastFM  
- Goodreads  

---

## 📈 Metrics
- Utility: Recall, Precision, MRR, NDCG  
- Fairness: ED, WGER, GC  

---

## 🔒 Note
This repository is anonymized for double-blind review.
