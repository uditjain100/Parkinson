# 🧠 Parkinson’s Disease Detection using CatBoost and XGBoost

Parkinson’s disease is a progressive neurological disorder that affects movement, speech, and posture. One of the earliest symptoms often seen in patients is vocal impairment, such as monotonic speech, tremors in voice, and reduced volume. These voice abnormalities provide an opportunity for early diagnosis using voice-based biomarkers. This project utilizes a dataset of vocal recordings from individuals with and without Parkinson’s disease to develop machine learning models that can accurately classify whether a patient is likely to be affected.

The primary goal of this project is to detect Parkinson's disease at an early stage using modern ensemble learning techniques. We use two powerful gradient boosting algorithms — **CatBoost** and **XGBoost** — to analyze vocal features and predict the presence of Parkinson’s. These models are trained on a preprocessed version of the dataset, where class imbalance is addressed using **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure robust prediction of the minority class. The data includes features like frequency variations (jitter), amplitude perturbations (shimmer), and noise-to-harmonics ratios, which are significant indicators of vocal degradation.

This approach is highly beneficial in the medical domain, especially in telemedicine, as it allows for non-invasive, cost-effective, and early-stage detection of the disease through voice samples. The project not only demonstrates the application of machine learning in healthcare but also emphasizes the importance of data preprocessing, feature scaling, and model evaluation using metrics such as Accuracy and Matthews Correlation Coefficient (MCC). Our final models achieve high classification performance, with CatBoost providing the best results.

---

## ✅ Key Features of the Project

1. **Early Detection**: Predicts Parkinson’s disease before severe symptoms appear using voice biometrics.
2. **High Accuracy**: Achieved up to **96.6% accuracy** using CatBoost with fine-tuned hyperparameters.
3. **Robust Metrics**: Evaluates model performance using **Matthews Correlation Coefficient (MCC)**, a balanced measure even for imbalanced classes.
4. **SMOTE Oversampling**: Balances the dataset by synthetically creating minority class samples.
5. **Feature Normalization**: Applies Min-Max scaling to bring all features to a common scale.
6. **Gradient Boosting Models**: Leverages XGBoost and CatBoost, known for high speed and accuracy in tabular data.
7. **GridSearchCV Tuning**: Optimizes hyperparameters systematically using cross-validation.
8. **Visualizations**: Includes confusion matrix, feature importance, and model performance graphs.
9. **Voice-based Input**: Utilizes voice-related biomedical features such as jitter, shimmer, NHR, and HNR.
10. **Scalable Workflow**: The notebook can be extended to other biomedical voice datasets for similar classification problems.

---

## 📂 Dataset Description

The dataset includes 195 samples and 24 features such as:

- `MDVP:Fo(Hz)`: Average vocal fundamental frequency
- `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`: Max/Min vocal frequencies
- Jitter and Shimmer features
- `NHR`, `HNR`: Noise-to-harmonics ratios
- `status`: Target variable (1 = Parkinson’s, 0 = Healthy)

Source: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

---

## 🛠️ Model Pipeline

1. **Data Cleaning & Normalization**  
   The dataset was first examined for missing values or inconsistencies. Since the data did not contain null entries, we proceeded with feature scaling. Each numeric feature was normalized using **Min-Max Scaling** to transform values into a range of [0, 1]. This ensured that all features contributed equally to the model and avoided dominance by features with larger numerical ranges.

2. **Class Imbalance Handling**  
   The dataset was imbalanced, with more Parkinson’s-positive samples than healthy ones. To correct this, we used **SMOTE (Synthetic Minority Over-sampling Technique)** from the `imbalanced-learn` library. SMOTE generates synthetic data points for the minority class by interpolating between existing samples, which helped in building models that could better generalize and not bias towards the majority class.

3. **Model Training**  
   We trained three classification models:

   - **XGBoost**: An efficient gradient boosting framework that builds trees in parallel and handles missing data internally.
   - **CatBoost**: A high-performance library by Yandex optimized for categorical features and known for its excellent out-of-the-box performance.
   - **Random Forest**: A bagging ensemble technique that constructs multiple decision trees and uses majority voting for predictions.

   For each model, we applied **GridSearchCV** for hyperparameter tuning. This process involved defining a parameter grid and performing exhaustive search using cross-validation to find the best parameter combinations that yielded the highest performance.

4. **Evaluation Metrics**  
   After training, we evaluated all models using the following metrics:
   - **Accuracy**: Measures the overall correctness of the model by calculating the ratio of correct predictions to total predictions.
   - **MCC (Matthews Correlation Coefficient)**: A robust metric that accounts for true and false positives and negatives. It provides a balanced evaluation even for imbalanced datasets.
   - **Confusion Matrix**: A matrix showing the counts of true positive, false positive, true negative, and false negative predictions, helping to visually inspect where the model is making errors.

---

## 🔍 Results

The performance of each model was measured using two key evaluation metrics: **Accuracy** and **Matthews Correlation Coefficient (MCC)**. Accuracy gives a general idea of how often the model predicts correctly, while MCC offers a more reliable measure for imbalanced datasets, as it considers all four components of the confusion matrix (TP, TN, FP, FN).

- The **CatBoost** model outperformed others with an **accuracy of 96.6%** and an **MCC of 91.4%**, making it the most reliable for predicting both Parkinson-positive and healthy samples. Its superior handling of categorical features and efficient default parameter settings contributed to this result.

- The **XGBoost** model came in second, achieving **94.9% accuracy** and **87.4% MCC**. While still highly effective, it required more extensive hyperparameter tuning to approach the performance levels of CatBoost.

- The **Random Forest** model showed **93.2% accuracy** and **82.4% MCC**. Although it is a robust baseline model, it lacked the boosting advantage and struggled slightly more with the imbalanced dataset compared to the gradient boosting models.

These results indicate that gradient boosting algorithms, especially CatBoost, are highly effective for early Parkinson’s disease detection using voice data. MCC scores above 90% also confirm that the models are not only accurate but balanced in predicting both classes.

| Model         | Accuracy (%) | MCC (%)  |
| ------------- | ------------ | -------- |
| CatBoost      | **96.6**     | **91.4** |
| XGBoost       | 94.9         | 87.4     |
| Random Forest | 93.2         | 82.4     |

---

## 📦 Requirements

```txt
pandas==2.0.3
numpy==1.24.4
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
catboost==1.2
imbalanced-learn==0.11.0
jupyter==1.0.0
```

---

## 🖥️ How to Run

Follow the steps below to set up and run the Parkinson’s Disease Detection project on your local machine:

1. **Clone the Repository**  
   First, clone the GitHub repository to your local system using the following command:

   ```bash
   git clone https://github.com/your-username/parkinsons-prediction.git
   ```

2. **Navigate to the Project Directory**  
   Change into the project directory:

   ```bash
   cd parkinsons-prediction
   ```

3. **Install Required Dependencies**  
   Ensure that Python (>=3.8) is installed. Then install all required Python packages using:

   ```bash
   pip install -r requirements.txt
   ```

   This will install packages like `pandas`, `numpy`, `xgboost`, `catboost`, `scikit-learn`, and `imbalanced-learn`.

4. **Launch the Jupyter Notebook**  
   Start Jupyter Notebook to open and execute the code:

   ```bash
   jupyter notebook parkinsons.ipynb
   ```

5. **Run the Notebook Cells**  
   In the opened notebook, execute each cell step-by-step (Shift + Enter) to:

   - Load and preprocess the dataset
   - Balance the dataset using SMOTE
   - Train the machine learning models
   - Evaluate and compare model performance

6. **Visualize and Interpret Results**  
   The notebook includes graphs, confusion matrices, and printed metrics to help interpret how well the models performed.

> **Note:** Ensure you have access to internet during dependency installation and Jupyter is installed in your environment. Use `pip install jupyter` if not already installed.

---

## 👩‍💻 Authors

- **Udit Jain**  
  B.Tech CSE Student, Maharaja Agrasen Institute of Technology (MAIT), Rohini, Delhi  
  Roll No: 10214802718  
  Email: jain30udit@gmail.com

- **Umang Tiwari**  
  B.Tech CSE Student, Maharaja Agrasen Institute of Technology (MAIT), Rohini, Delhi  
  Roll No: 10314802718  
  Email: umangtiwari2604@gmail.com

- **Himani Sheoran**  
  B.Tech CSE Student, Maharaja Agrasen Institute of Technology (MAIT), Rohini, Delhi  
  Roll No: 02514802718  
  Email: sheoran.himani@gmail.com

- **Zameer Fatima** _(Project Supervisor)_  
  Assistant Professor, Department of Computer Science and Engineering  
  Maharaja Agrasen Institute of Technology (MAIT), Rohini, Delhi  
  Email: zameerfatima@mait.ac.in

---
