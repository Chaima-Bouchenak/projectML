# ~ MACHINE LEARNING PROJECT ~
Use different ML algorithns on real gene expression RNS-seq Dataset for the Preciction of the type of cancer .

## Project description

In this project, 4 machine learning models to be used for cancer prediction were constructed using genes that encode human proteins 20531. More precisely, we predict the class of 5 cancers LUAD, COAD, PRAD, KIRC and BRCA, respectively breast cancer, kidney and kidney cancer, colorectal cancer, lung cancer and prostate cancer according to the level of expression of these genes in 801 individuals (RNA-Seq).

Implemented and used on the Pima Indians Diabeties Data set to predict whether or not the patients in the dataset have diabetes.


## Data descreption

The dataset we will use in our Project is the gene expression cancer RNA-Seq Data Set . 

This dataset is originally from the University of Genoa. This collection is part of the RNA-Seq (HiSeq) PANCAN data set, a random extraction of gene expressions of patients having different types of tumor (details below).

The objective of the dataset is to predict which kind of cancer a patient has, based on certain gene expression measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. 

You can learn more about this dataset description at the UCI Data Repository : https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

==> "data.csv": genetic expression of 20531 genes of 800 patients.
==> "labels.csv": the type of cancer of 800 patients.

There is 5 types of cancer : 

-> BRCA: Breast Invasive Carcinoma 

-> LUAD: Lung Adenocarcinoma

-> PRAD: Prostate Adenocarcinoma

-> KIRC: Kidney Renal Clear Cell Carcinoma

-> COAD: Colon Adenocarcinoma


## uplaoding the project

1- Press code (top right button) and select Download ZIP
Once the repository is downloaded unzip the projectML.zip folder and extract it to the folder you want.
2- Install a conda envirement in you local machine if it's not aleady done.
3- Creat the environment with conda :

Open the terminal and run the following command

with .yml file :

```
conda env create --file install/env.yml -n project_env

conda activate project_env
```
or with the requirement.txt file :

```
pip install -r requirements.txt (to install all dependencies)
```


## Running
------------------------
On terminal run : 
```
python main.py
```


## The algorhitms that was tested are : 

### Random Forest Classifier


### Logistic Regression (LR)


Logistic regression is used to predict the categorical placement of the different types of cancer from the TCGA project. I created the Logistic_Regression.py script to predict the accuracy score of the classification of the different types of cancer of the TCGA project 

### GaussianNB (NB)

### Decision Trees Classifier (CART)

Decision tree (DT) is a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the characteristics of TCGA data that contains discrete target variables. for that I implemented a Decision_Tree.py script

### Support Vector Machine (SVM)


