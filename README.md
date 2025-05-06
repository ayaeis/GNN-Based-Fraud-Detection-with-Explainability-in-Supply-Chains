# Graph Neural Networks for Supply Chain Fraud Detection with Explainability

This repository contains the full implementation of our research project titled **"Graph Neural Networks for Supply Chain Fraud Detection with Explainability via Saliency Maps and RL-based Feature Masking"**, conducted as part of the Department of Computer Science and Engineering at the American University of Sharjah.

Our project presents a novel approach to identifying fraudulent transactions in supply chains using **heterogeneous Graph Neural Networks (GNNs)** with a strong emphasis on **model explainability**.

---

## Project Overview

Fraud in supply chains often manifests through subtle, interconnected patterns involving customers, orders, and products. Traditional methods overlook these patterns due to their flat or rule-based nature.

This project builds a **heterogeneous GNN pipeline** on top of a **tripartite graph** (customer, order, product), trained on a simulated version of the **DataCo Supply Chain dataset** with fraud labels generated using realistic, rule-based heuristics.

To overcome the "black box" nature of GNNs, we integrate two complementary explainability methods:

* **Saliency Maps** (Gradient-based)
* **RL-based Feature Masking Explainer**

---

## Repository Structure

```
.
├── fraud_detection_gnn_with_explainer.ipynb   # Main notebook
├── README.md                                  # Project description
└── Graph_Neural_Networks_for_Supply_Chain_Fraud_Detection_with_Explainability_via_Saliency_Maps_and_RL_based_Feature_Masking.pdf
```

---

## Technologies Used

* Python 3.9+
* PyTorch & PyTorch Geometric
* NetworkX
* Scikit-learn
* Matplotlib / Seaborn
* NumPy / Pandas

---

## Methodology

### Dataset

* **Source**: DataCo Smart Supply Chain Dataset (Kaggle)
* **Structure**: Synthetic, relational, and realistic
* **Entities**: Customers (2,072), Orders (9,351), Products (1,859)

### Graph Construction

* **Node Types**: `Customer`, `Order`, `Product`
* **Edge Types**: `Customer → Order`, `Order → Product`
* **Features**: Behavioral, geographic, temporal, discount-based, and derived metrics (e.g., profit margin, unit discount)

### Fraud Labeling

Fraudulent behavior was labeled using domain-specific heuristics (e.g., excessive discounts, unreasonable delays, delivery mismatches), resulting in a fraud rate of **\~7.2%**.

---

## Model Architecture

* **Model**: 2-layer heterogeneous `GraphSAGE`
* **Framework**: PyTorch Geometric
* **Input**: Order node classification task (binary fraud detection)
* **Message Passing**: Type-specific with `HeteroConv`
* **Evaluation Metrics**:

  * Accuracy: 97.10%
  * Macro F1-Score: 88.28%
  * Weighted F1-Score: 96.96%
  * Precision (Fraud): 86.84%
  * Recall (Fraud): 70.97%

---

## Explainability

### 1. **Gradient-Based Saliency Maps**

* Captures **global feature importance**
* Operates on order, customer, and product features
* Reveals high influence from:

  * Shipping delays
  * Profit margins
  * Discount rates
  * Geographic and market indicators

### 2. **RL-Based Feature Masking**

* Provides **sparse, instance-specific explanations**
* Trained using REINFORCE to preserve GNN decisions under feature masking
* Learns minimal feature subsets per node type
* Yields insights into **localized fraud patterns**

> Together, these techniques provide both **global trust** and **local justifications**.

---

## Results Summary

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 97.10% |
| Macro F1-Score    | 88.28% |
| Weighted F1-Score | 96.96% |
| Fraud Precision   | 86.84% |
| Fraud Recall      | 70.97% |
| Fraud F1-Score    | 78.11% |

---

## Key Contributions

* Designed a **tripartite, heterogeneous graph** for supply chain modeling
* Developed an interpretable **GNN pipeline** that preserves relational integrity
* Integrated a **dual-layer explainability** module with saliency and RL-based feature masking
* Achieved **high performance** while maintaining **interpretability**

---

## Limitations

* Dataset is synthetic, not operational
* Fraud labels are heuristic-based, not ground-truth
* Explainability tools tailored for homogeneous graphs were not directly usable

---

## Future Work

* Adapt to real-world fraud datasets with confirmed labels
* Integrate temporal or sequential GNN models (e.g., TGAT)
* Extend explainability methods to **edge importance**
* Deploy model for real-time fraud detection in enterprise systems

---

## Citation

If you use this project or build on it, please cite:

```
Aieh Eissa, Tsion Regasa. "Graph Neural Networks for Supply Chain Fraud Detection with Explainability via Saliency Maps and RL-based Feature Masking." American University of Sharjah, 2025.
```

---

## Contact

For questions or collaborations:

* **Aieh Eissa** – [g00107537@aus.edu](mailto:g00107537@aus.edu)
* **Tsion Regasa** – [g00107807@aus.edu](mailto:g00107807@aus.edu)

