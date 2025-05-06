# GNN-Based-Fraud-Detection-with-Explainability-in-Supply-Chains
# 📦 GNN-Based Fraud Detection with Explainability in Supply Chains

This repository presents our research project on using **Graph Neural Networks (GNNs)** for detecting fraud in supply chain networks, enhanced with **explainability tools** such as saliency maps and reinforcement learning-based feature masking. Our approach targets subtle and distributed fraud patterns in procurement, logistics, and financial flows.

---

## 📁 Project Structure


---

## 📌 Overview

Supply chains are vulnerable to various types of fraud including:
- Fake vendors in procurement
- Duplicate financing of receivables
- Manipulated shipment records

These patterns are often **non-linear, sparse, and context-dependent**, making them ideal for graph-based approaches. We propose a **hybrid model combining heterogeneous GNNs ** with **explainability tools** to improve both accuracy and trust.

---

## 🚀 Key Components

### 🧠 Model Architecture
- **Heterogeneous Graph Neural Network**: Encodes multi-entity relationships (supplier → order → customer).
- **Reinforcement Learning Agent**: Learns to mask non-informative features to improve interpretability.
- **Saliency Mapping**: Identifies most influential nodes and edges in the fraud decision.

### 🛠 Tools Used
- PyTorch Geometric (PyG)
- DGL (optional extensions)
- Captum (for saliency maps)
- RLlib (for reinforcement learning agent)

---

## 📊 Results Summary

| Model          | Macro-F1 | AUC-ROC | Explainability Tool | Remarks                         |
|----------------|----------|---------|----------------------|----------------------------------|
| MultiFraud     | 0.81     | 0.89    | Saliency Maps        | High precision, hard to trace   |
| FraudGNN-RL    | 0.79     | 0.91    | RL-Based Masking     | Slightly lower F1, better trust |
| CNN Baseline   | 0.62     | 0.74    | —                    | Ignores structural patterns     |

---

## 🎥 Demo Videos

- 📊 **Supervised Training**  
  *Model learning entity embeddings and edge classification*  
  [Watch Training Demo](videos/training_demo.mp4) *(local)*

- 🤖 **Reinforcement Learning in Action**  
  *Feature masking agent optimizing fraud signal*  
  [Watch RL Demo](videos/rl_learning.mp4)

---

## 📄 Report and Slides

- 📘 [Final Report (PDF)](report/final_report.pdf)
- 📽️ [Presentation Slides](presentation/gnn_fraud_detection_slides.pdf)

---

## ⚙️ Setup Instructions

### 🔧 Environment

Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
