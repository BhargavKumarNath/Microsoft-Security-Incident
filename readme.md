# Microsoft Security Incident Prediction

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![alt text](https://img.shields.io/badge/Pandas-2.0-blue?style=for-the-badge&logo=pandas)
![alt text](https://img.shields.io/badge/LightGBM-4.0-blue?style=for-the-badge&logo=lightgbm)
![alt text](https://img.shields.io/badge/XGBoost-2.0-blue?style=for-the-badge&logo=xgboost)
![alt text](https://img.shields.io/badge/PyTorch-2.0-blue?style=for-the-badge&logo=pytorch)
![alt text](https://img.shields.io/badge/Transformers-Custom-blue?style=for-the-badge&logo=huggingface)
![alt text](https://img.shields.io/badge/scikit--learn-1.3-blue?style=for-the-badge&logo=scikit-learn)
![alt text](https://img.shields.io/badge/SHAP-Explainable%20AI-blue?style=for-the-badge)

An end to end Machine Learning project to predict the triage grade of real world cybersecurity incidents using Microsoft's GUIDE dataset. This repository showcase a comprehensive workflow, from large scale data processing and advanced feature engineering to comparative model benchmarking and deep insights.

# 1. Introduction
In today's cybersecurity landscape, Security Operations Centers (SOCs) are inundated with a massive volutme of security alerts. The sheer number of potential threats makes it impossible for human analysis to manually investigate every single one. This "alert fatigue" can lead to critical threats being missed.

Machine Learning offers a powerful solution by automating the initial triage process. By learning from historical data,an ML model can access incoming security incidents and assign a preliminary grade, allowing analysts to focus their limited time and resources on the events that are most likely to be genuine, melivious attacks (`True Positives`).

This project tackles this challange head on by building a robust system to classify security incidents, demonstrating a data driven approach to enhancing cybersecurity operations.

# 2. Dataset Description
This project utilises the GUIDE (Guided Response Investigation Dataset), the largest publicly available collection of real world cybersecurity incidents, released by Microsoft.

**Key characteristics:**
    
* **Scale:** Over 13 million pieces of evidence across 1 milltion triage annotated incidents.
* **Diversity:** Telemetry from over 6,100 organisations, featuring 9,100 unique build in and custom `DetectorIds`.
* **Hierarchical Structure:** The data is organised in three levels:

    1. **Evidence:** The lowest level, representing individual logs or entities (e.g., an IP address, a file hash, a user account).
    2. **Alert:** A consolidation of one or more pieces of evidence that signifies a potential security event.
    3. **Incident:** The highest level, grouping one or more related alerts into a cohesive security narrative that requires triage.
* **Privacy:** All sensitive data has been rigorously anonymized through a multi stage process of hasing, randomization, and timestamp perturbation to protect privacy while preserving data utility.

# 3. Project Objective
The primaru objective of this project is to develop a Machine Learning model that accurately predicts the `IncidentGrade` for a given 