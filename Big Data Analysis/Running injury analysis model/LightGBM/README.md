# Method

This project builds a sports injury prediction model based on high-level track and field training data, and explores the differences in the effects of traditional machine learning and deep time series modeling methods in unbalanced medical data. LightGBM and TCN are used for sports injury prediction. LightGBM is an efficient ensemble algorithm based on decision trees, suitable for structured feature learning; TCN effectively captures long-range dependencies in time series through dilated causal convolution, improving the model's ability to model changes in training load. In order to address the serious imbalance problem in the data, SMOTE is used to generate minority class samples, and ENN is combined to remove noise points, thereby optimizing the model's discrimination boundary and generalization performance.



# Dataset Description

**Source:** A national middle-distance running team of the Netherlands (2012 - 2019).

**Participants:** 74 athletes (27 female, 47 male), data collected by the same head coach, with a unified training system.

**Event Overview:** 42183 health records and 583 injury records (accounting for only about 1.4%, with a serious imbalance).

**Download:** https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/UWU9PV

**Structure:** Organize the dataset as follows:

```
dataset
   |--- day_approach_maskedID_timeseries.csv
   |--- week_approach_maskedID_timeseries.csv
```



# Installation

## Requirements

- lightgbm==4.6.0
- scikit-learn==1.6.1
- imbalanced-learn==0.13.0
- pandas==2.2.3
- numpy==1.24.3
- matplotlib==3.10.3
- joblib==1.5.0
- torch==2.4.1

## Setup

For **LightGBM**, choose "**lightgbm.conda**" to set the environment.

Run this command to install environments:

```
pip install -r requirements.txt
```

Run this command to install ipykernel:

```
pip install ipykernel
```

For **TCN**, run this command to install environments:

```
conda env create -f tcn_model.yaml
```



# Evaluation

For **LightGBM**, use **Cell > Run All** from the menu, or run cell sequentially using **Shift + Enter**.

For **TCN**, use this command:

```
python train.py
```



# Results

| Model    | Precision | Recall | F1     | AUC    |
| -------- | --------- | ------ | ------ | ------ |
| LightGBM | 0.2632    | 0.0855 | 0.1290 | 0.7254 |
| TCN      | 0.9697    | 0.9057 | 0.9366 | 0.9997 |

