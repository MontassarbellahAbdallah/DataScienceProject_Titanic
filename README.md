# Data Analysis and Classification Project

This project demonstrates a complete pipeline for data analysis and classification using various machine learning algorithms. 
The primary focus is on exploring the dataset, preparing the data, and training multiple classifiers to predict outcomes, including K-Nearest Neighbors (K-NN), 
Support Vector Machines (SVM), Logistic Regression, and Random Forests, are trained using cross-validation to evaluate their performance in predicting outcomes. 
The project emphasizes handling missing data, standardizing features, and using visualization tools like Matplotlib and Seaborn to enhance data understanding. 
The results from each model are compared to identify the best-performing algorithm.


## Features

- **Data Visualization**: Utilizes Matplotlib and Seaborn for creating insightful visualizations.
- **Data Imputation**: Handles missing data using K-Nearest Neighbors (KNN) imputation.
- **Feature Scaling**: Applies Min-Max scaling to standardize the feature set.
- **Model Training**: Trains multiple classifiers (K-NN, SVM, Logistic Regression, Random Forests) using cross-validation.
- **Model Comparison**: Compares the performance of different models to find the best-performing one.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MontassarbellahAbdallah/DataScienceProject_Titanic.git
2. Navigate to the project directory:
  '''bash
  cd data-analysis-classification

3. Install the required dependencies:

## Usage
Load the Data: The dataset should be placed in the project directory, named train.csv and test.csv.
Run the Analysis: Execute the script to perform data analysis, visualization, and model training.

python main.py
View Results: The script outputs the performance of each classifier and provides visualizations for data exploration.

## Dependencies
Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn

You can install these dependencies using:
pip install pandas numpy scikit-learn matplotlib seaborn

## License
This project is not for comercial use.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Acknowledgments
This project was inspired by the need to effectively classify and predict outcomes using machine learning techniques.
