# PRODIGY_TrackCode_TaskNumber1
# 🏠 House Price Prediction using Machine Learning

This project builds a **House Price Prediction model** using Machine Learning techniques.  
The model predicts housing prices based on important property features such as living area, house quality, number of bathrooms, and garage capacity.

This project was completed as part of my **Machine Learning Internship Task at Prodigy Infotech**.

---

# 📌 Project Objective

The goal of this project is to build a **regression model** that predicts house prices using housing features from the dataset.

The project demonstrates a complete **Machine Learning workflow**, including:

- Data loading
- Data exploration
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Data visualization

---

# 📂 Dataset

Dataset used:

**House Prices – Advanced Regression Techniques**

The dataset contains detailed information about residential homes.

### Key Features Used

| Feature | Description |
|------|------|
| GrLivArea | Above ground living area (square feet) |
| OverallQual | Overall material and finish quality |
| GarageCars | Garage capacity |
| TotalBsmtSF | Total basement area |
| FullBath | Number of full bathrooms |
| YearBuilt | Year the house was built |
| TotRmsAbvGrd | Total rooms above ground |
| HouseAge | Engineered feature calculated from YearBuilt |

### Target Variable

**SalePrice** – The selling price of the house.

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

---

# 🧠 Machine Learning Model

This project uses **Ridge Regression**, which is a regularized version of Linear Regression.

### Why Ridge Regression?

Ridge regression helps:

- Reduce overfitting
- Improve model generalization
- Handle multicollinearity between features

The machine learning pipeline includes:

1. Feature scaling using **StandardScaler**
2. Regression using **Ridge Regression**

---

# 🔄 Machine Learning Pipeline

The workflow implemented in this project:

1. Load the dataset
2. Perform data exploration
3. Handle missing values
4. Remove outliers
5. Perform feature engineering
6. Split dataset into training and testing sets
7. Train the machine learning model
8. Evaluate model performance
9. Visualize predictions

---

# 📊 Model Evaluation

The model is evaluated using the following metrics:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Typical performance achieved:

```
R² Score ≈ 0.80 – 0.90
```

This indicates the model explains a significant portion of the variance in house prices.

---

# 📈 Visualizations

The project includes several visualizations to understand the data and model performance:

- Feature correlation heatmap
- Actual vs Predicted price scatter plot

These visualizations help analyze relationships between variables and model predictions.

---

# 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/kirtiarya05/PRODIGY_TrackCode_TaskNumber1.git
```

### 2️⃣ Navigate to the project directory

```bash
cd Task1
```

### 3️⃣ Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4️⃣ Run the project

```bash
main.py
```

---

# 📁 Project Structure

```
house-price-prediction
│
├── train.csv
├── main.py
├── README.md
```

---

# 🎯 Key Learning Outcomes

Through this project, I learned:

- Data preprocessing techniques
- Feature engineering
- Regression models in machine learning
- Model evaluation metrics
- Building a complete ML pipeline
- Visualizing insights from data

---

# 👨‍💻 Author

**Kirti**

Machine Learning Intern  
Prodigy Infotech Internship

---

# ⭐ Future Improvements

Possible improvements for this project:

- Using more features from the dataset
- Trying advanced models such as Random Forest or XGBoost
- Hyperparameter tuning
- Deploying the model as a web application
