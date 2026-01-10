# Smart Crop Recommendation System 🌾

## 📌 Project Overview
Smart Crop Recommendation System is an AI-based tool that helps farmers in Pakistan choose the most suitable and profitable crop using soil and weather data such as pH, rainfall, temperature, and humidity. It supports data-driven farming to improve yield, reduce risk, and promote sustainable agriculture.

This project is built from scratch without using pre-trained deep learning models, focusing on the fundamental mathematics of Machine Learning algorithms.

**Student:** Shah Zaib Khan (Student ID: SU92-MSAIW-S25-019, MS-AI)
**Teammate:** Talha Bin Aslam (Student ID: SU92-MSAIW-F25-043, MS-AI)
**Project Type:** Machine Learning / Classification (Custom Model)

---

## 📅 Weekly Progress Log

### ✅ Week 1: Foundation, Data Pipeline & Exploration
**Status:** Completed
**Key Activities:**
- **Dataset Acquisition:** Downloaded the "Crop Recommendation Dataset" from Kaggle (2200 rows, 22 crop types).
- **Data Preprocessing:**
  - Checked for missing values (dataset is clean).
  - Analyzed statistical properties of soil features (N, P, K, Temperature, etc.).
- **Visualization:**
  - Created a **Correlation Heatmap** to understand relationships between environmental factors.
- **Data Pipeline:**
  - Implemented an **80-20 Train-Test Split** to prepare data for modeling.
  - Verified the shape of Training (1760 samples) and Testing (440 samples) sets.
- **Documentation:**
  - Uploaded the initial project proposal and timeline.

---

### ✅ Week 2: Model Development & Stabilization
**Status:** Completed
**Key Activities:**
- **Model Architecture:**
  - Built a custom **Decision Tree Classifier** using Scikit-Learn (Entropy criterion).
  - Set `max_depth=12` to balance accuracy and prevent overfitting.
- **Training & Evaluation:**
  - Trained the model on the training dataset (1760 samples).
  - Achieved a high validation accuracy of **97.95%** on the test set.
- **Model Stabilization:**
  - Implemented `random_state=42` to ensure 100% reproducible results every time the code is run.
  - **Model Persistence:** Saved the trained model as a `.pkl` file (`crop_recommendation_model.pkl`) to lock the learned weights permanently.
- **Visualization:**
  - Generated a decision tree plot to visualize the decision-making logic of the AI.

![Decision Tree Logic](decision_tree_visual_clean.png)
*Figure 1: Visualization of the Decision Tree Logic (Top 3 Levels)*

- **Deliverables:**
  - Uploaded the trained `.pkl` model file to the repository.


  ### ✅ Week 3: Inference Engine & Stress Testing
**Status:** Completed
**Key Activities:**
- **Model Deployment (Simulation):**
  - Successfully loaded the `.pkl` file using `joblib` to simulate a production environment.
- **Inference Function:**
  - Developed a `recommend_crop()` function that accepts 7 input parameters (N, P, K, Temp, Hum, pH, Rain).
  - **Optimization:** Refactored the function to use Pandas DataFrames, resolving Scikit-Learn feature warning issues.
- **Unit & Stress Testing:**
  - Tested the system with varied "Farmer Scenarios" to validate logic.
  - **Normal Conditions:** Correctly predicted **Rice** for wet/high-nitrogen soil and **Chickpea** for dry soil.
  - **Stress Testing (Edge Cases):**
    - *Cold Climate (12°C):* Predicted **Grapes** (Correct).
    - *Poor Soil (Low Nutrients):* Predicted **Mothbeans** (Correct).
    - *Tropical Swamp (High Heat/Humidity):* Predicted **Coconut** (Correct).

---

## 🛠 Tools & Technologies
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
- **Environment:** Google Colab / Jupyter Notebook

### ✅ Week 4: Performance Evaluation & Documentation
**Status:** Completed
**Key Activities:**
- **Rigorous Evaluation:**
  - Generated a comprehensive Classification Report (Precision, Recall, F1-Score).
  - Achieved an overall **Accuracy of 98%**.
  - Identified **Perfect Predictions (F1=1.0)** for crops like Apple, Banana, and Cotton.
- **Error Analysis:**
  - Visualized errors using a **Confusion Matrix**.
  - Observed minor confusion between **Rice** and **Jute**, likely due to their similar high-water requirements.
- **Documentation:**
  - Finalized the project report with a literature review of recent studies.
  - Organized the repository for submission.

![Confusion Matrix](confusion_matrix.png)
*Figure 2: Confusion Matrix showing model performance across all 22 crop classes.*
