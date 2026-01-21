# Network Intrusion Detection Using CICIDS-2018 Dataset

## Project Overview
This project implements a machine learning–based Network Intrusion Detection System using the CICIDS-2018 dataset provided by the Canadian Institute for Cybersecurity.

The objective is to classify network traffic as either benign or malicious by learning patterns present in real-world network data.

This project was developed as part of the AI & Machine Learning academic coursework.

---

## Dataset Description
The CICIDS-2018 dataset contains realistic network traffic generated in a controlled environment and includes both benign and attack activities.

Attack categories present in the dataset include:
- Brute Force attacks
- Denial of Service (DoS)
- Distributed Denial of Service (DDoS)
- Botnet traffic
- Infiltration attacks
- Web-based attacks

Due to the large size of the dataset, the dataset files are not included in this repository.

Dataset source:
https://www.unb.ca/cic/datasets/ids-2018.html

---

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Methodology
The project follows a structured machine learning workflow:

1. Data loading and understanding  
2. Data preprocessing and cleaning  
3. Handling missing and infinite values  
4. Feature normalization and scaling  
5. Feature selection  
6. Splitting data into training and testing sets  
7. Model training using Support Vector Machine  
8. Hyperparameter tuning using Random Search and Grid Search  
9. Model evaluation  

---

## Machine Learning Model
### Support Vector Machine (SVM)

Support Vector Machine was used as the primary classifier due to its effectiveness in high-dimensional feature spaces.

To improve performance:
- **RandomizedSearchCV** was applied to explore a broad hyperparameter range efficiently.
- **GridSearchCV** was then used for fine-tuning optimal parameters.

This approach helped in achieving better generalization and reduced model overfitting.

---

## Evaluation Metrics
Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics were used to assess the intrusion detection capability of the model.

---

## Results
The Support Vector Machine model was evaluated using both Randomized Search and Grid Search for hyperparameter tuning.

Based on the experimental results presented in the report, Randomized Search achieved better classification performance compared to Grid Search.  
Random Search was able to identify more effective hyperparameter combinations within a limited search space, leading to improved evaluation metrics.

The results demonstrate that Randomized Search was more efficient and better suited for tuning the SVM model on the CICIDS-2018 dataset.

---

## Key Learnings
- Working with real-world cybersecurity datasets
- Importance of data normalization for SVM models
- Hyperparameter tuning using Random Search and Grid Search
- Evaluating classifiers using multiple performance metrics
- Understanding intrusion detection systems

---

## Future Scope
- Experiment with deep learning–based intrusion detection models
- Implement real-time network traffic monitoring
- Improve computational efficiency for large-scale data
- Deploy the system as a security analytics application

---

## Author
**Mansi Behera**  
MCA, National Institute of Technology, Raipur


## Project Structure
