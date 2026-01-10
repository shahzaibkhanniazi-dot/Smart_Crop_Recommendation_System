# Smart Crop Recommendation System 🌾

## 📌 Project Overview
Smart Crop Recommendation System is an AI-based tool that helps farmers in Pakistan choose the most suitable and profitable crop using soil and weather data such as pH, rainfall, temperature, and humidity. It supports data-driven farming to improve yield, reduce risk, & promote sustainable agriculture.

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

---
### 🔬 Advanced Research Validation
To ensure the model is scientifically robust, we performed 5 rigorous stress tests:

**1. Stability Test (5-Fold Cross-Validation)**
- **Method:** Split data into 5 different subsets and trained 5 separate times.
- **Result:** Average accuracy remained consistent at **~98%**, proving the model is stable and not just "lucky" with one specific dataset.

**2. Interpretability (Feature Importance)**
- **Method:** Analyzed which soil factors contribute most to the decision.
- **Insight:** As shown in the graph below, **Potassium (K)** and **Humidity** were identified as the dominant features, aligning with agronomic science (water and root development are critical).

![Feature Importance Graph](feature_importance.png)
*Figure 3: Feature Importance analysis showing Potassium and Humidity as the primary decision drivers.*

**3. Robustness Check (Noise Injection)**
- **Method:** Injected random noise (+/- 5%) into input data to simulate faulty sensors.
- **Result:** The model maintained correct predictions (e.g., still predicting "Rice" even with noisy data), demonstrating robustness against real-world data imperfections.

**4. Baseline Comparison (The "Sanity Check")**
- **Method:** Compared our Decision Tree against a "Dummy Classifier" (Random Guessing).
- **Result:**
  - Dummy Model Accuracy: **~4.5%**
  - Our AI Model Accuracy: **97.95%**
  - **Conclusion:** The AI is significantly learning patterns, not just guessing.

**5. Learning Curve Analysis**
- **Method:** Plotted accuracy vs. training size.
- **Result:** The curve below shows that the Cross-Validation Score (Green Line) increases steadily as we add more data, eventually converging with the Training Score. This indicates that our dataset size (2200 samples) is sufficient for this problem.

![Learning Curve Graph](learning_curve.png)
*Figure 4: Learning Curve demonstrating model improvement and convergence with increased data size.*

## 🔮 Future Scope & Improvements
While this project successfully demonstrates the power of the Decision Tree algorithm for crop recommendation, there are several areas for future enhancement:
1. **Web Interface:** Developing a user-friendly frontend using **Gradio** or **Streamlit** to allow farmers to easily input data without writing code.
2. **Deep Learning Integration:** Implementing **CNNs (Convolutional Neural Networks)** to detect plant diseases from leaf images, combining visual data with soil metrics.
3. **Real-Time IoT Data:** Integrating with IoT sensors to automatically fetch soil moisture and temperature data instead of manual entry.
4. **Localization:** Adding support for local languages (Urdu) to make the tool accessible to rural farmers in Pakistan.

---

## 👨‍💻 Author & Contact
**Project By:**
* **Shah Zaib Khan** (ID: SU92-MSAIW-S25-019)
* **Talha Bin Aslam** (ID: SU92-MSAIW-F25-043)

**Course:** MS-AI (Fall 2025)
**University:** Superior University
**Instructor:** Sir Talha Nadeem

---
*This project was built from scratch to demonstrate fundamental Machine Learning concepts without reliance on pre-trained APIs.*
