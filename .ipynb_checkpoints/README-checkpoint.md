# Predicting Material Properties using Machine Learning

This repository contains three Jupyter notebooks (`Shear_modulus_G.ipynb`, `Form_energy_Ef.ipynb` and `Band_gap_Eg.ipynb`) demonstrating the application of machine learning (ML) techniques in materials science. The data is sourced from repositories like [Matbench](https://hackingmaterials.lbl.gov/automatminer/datasets.html#accessing-the-ml-tasks%20MatBench%20v0.1%20benchmark%20%E2%80%94%20Automatminer%201.0.3.20200727%20documentation).


## 🎯 Objectives

The objective of this project is to use simple regression models and composition-based features to predict material properties such as:

*   Shear modulus
*   Formation energy and
*   Band gap


## Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- Basic knowledge of Python, pandas, scikit-learn, and matplotlib.

## 📚 Libraries
The notebook uses the following libraries:

- numpy
- pandas
- matplotlib
- pymatgen (for composition handling)
- CBFV (Composition-Based Feature Vector package: pip install CBFV)
- scikit-learn (for ML models and preprocessing)
- gdown (for downloading data from Google Drive)
- seaborn (for enhanced visualizations, optional)


## 📖 Dataset
The datasets (1-ByLMUHJUKJp659iZCwnAu6_QFy3nC1j for shear modulus - **matbench_log_gvrh.json**, 1bIP5dcQYdO2KpLCvbubYAToJFv55HPHu for formation energy- **fe_10000.csv**, 1ie91WlCkyZc-kYtPhaUlRWq9oZxGXPWU for band gap - **bg_10000.csv**) contain material compositions and their properties. It is downloaded using gdown from a Google Drive link.

*Columns*: ID, formula, target  &nbsp;&nbsp;&nbsp;  *Shape*: (10000, 3)

The shear modulus data was in JSON format and had to be converted to pandas dataframe.  
The formation energy and band gap data was in csv format.

## 🪜 STEPS:
1. **Dataset**: Downloaded the dataset in JSON format/csv format and converted to pandas DataFrame.
2. **NaN values**: Checked and handled missing (NaN) values by dropping them.
   
4. **Generated composition-based feature vectors**: To featurize the chemical compositions from a chemical formula (e.g. "Al2O3") into a composition-based feature vector (CBFV), we use the open-source CBFV package. The default methodology to generate the features is 'Oliynyk'.

5. **Split the data into training and testing sets**: We used 80% of the data for training and 20% of the data for testing.
6. **Scaling and normalizing**: We scaled and normalized the features using StandardScaler in scikit-learn library.
7. **Correlations and feature selection**: We constructed a correlation matrix (pearson) to analyze feature relationships. We select the features with correlations less than 0.95 to remove redundant features and get the independent features. Among these independent features we choose the ones that are moderately/highly correlated (>0.45) with the target.
8. **Visualization** We considered the target variable distribution.
9. **Models**: We trained and evaluated simple machine learning models.
10. **Prediction**: Plotted and compared predicted versus actual results.
11. **Hyperparameter-tuning**: Performed hyperparameter tuning for model optimization using GridSearchCV.

## ⚙️ ML Models used
Multiple regression models are trained and compared, including:
- <u> Ridge regression <u/>
- <u> Lasso regression <u/>
- <u> Random forest regressor <u/>
- <u> KNeighbours regressor <u/>
- <u> Support vector regressor <u/>

## 📈 Results

### ▶️ Correlations between features
The darkness of the cells represent higher correlations

<img src="gvrh_correlations.png" alt="drawing" width="550"/>

<br>

### ▶️ Shear Modulus
The distribution of the Shear Modulus over the loaded data-set

<img src="gvrh_distribution.png" alt="drawing" width="300"/>

The predicted shear modulus in comparison to the real values

<img src="gvrh_predict.png" alt="drawing" width="300"/>

<br>

### ▶️ Formation Energy
The distribution of the formation energies over the loaded data-set

<img src="form_distribution.png" alt="drawing" width="300"/>

The predicted formation energies in comparison to the real values

<img src="form_predict.png" alt="drawing" width="300"/>

<br>

### ▶️ Band-Gap
The distribution of the band-gaps over the loaded data-set

<img src="band_distribution.png" alt="drawing" width="300"/>

The predicted formation energies in comparison to the real values

<img src="band_predict.png" alt="drawing" width="300"/>


## 💡 Key Insights
✅ **Data Quality**: The dataset contains 10,000 samples with no missing values after cleaning  
✅ **Feature Engineering**: CBFV's Oliynyk features effectively represent chemical compositions  
✅ **Visualizations**: It helps assess model performance and data characteristics  
✅ **Model Performance**: Ensemble methods like Random Forest often outperform simpler models  
✅ **Hyperparameter Importance**: Default parameters can sometimes be optimal