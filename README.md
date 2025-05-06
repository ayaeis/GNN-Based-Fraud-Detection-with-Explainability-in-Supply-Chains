# GNN-Based Fraud Detection with Explainability in Supply Chains

This repository contains the implementation and supporting resources for the research project:  
**"Graph Neural Networks for Supply Chain Fraud Detection with Explainability via Saliency Maps and RL-based Feature Masking"** by Aieh Eissa and Tsion Regasa.

## 📌 Project Summary

Fraud in supply chains often arises from subtle, cross-entity interactions among customers, orders, and products. This project presents a heterogeneous graph-based approach for fraud detection using a two-layer GraphSAGE model. To address the black-box nature of graph neural networks, we incorporate two complementary explainability methods: gradient-based saliency analysis and a reinforcement learning-based feature masking agent.

## 💡 Key Features

- **Tripartite Heterogeneous Graph**: Customers → Orders → Products, preserving supply chain semantics.
- **GraphSAGE-based GNN**: Enables inductive learning over multi-entity transaction data.
- **Synthetic Supply Chain Dataset**: Fraud labels generated via domain heuristics.
- **Explainability Tools**:
  - *Gradient-based Saliency Maps*: Identifies feature sensitivity globally.
  - *RL-Based Feature Masking*: Produces sparse, local, human-readable explanations.
- **Evaluation**: Achieved 97.1% accuracy and 88.28% macro F1 on an imbalanced fraud dataset.

## 📂 Repository Structure


