# Heart Disease Prediction 
This is a python package for the purposes building a predictive model of the presence of heart disease using patient 
level healthcare data. 
Data source and information: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

### Step by step setup and run guide
1) Create environment with requirements.txt
2) If previously installed, run "pip uninstall heartDiseasePrediction" and then "python setup.py clean"
3) Run "python setup.py develop" to install package
4) Run "src/main.py" to read data, preprocess and generate results of baseline random forests model

### Package notes 22-02-21
Early development version of package including data pipeline, preprocessing and baseline random forest model assesment.

### Package development requirements
1) Documentation
2) Testing
3) Parsing of corrupted cleveland data

### Future experiments
1) Random forest hyper-parameter optimisation via cross_validation
2) Variable importance scoring
3) Boosted decision trees comparison to random forrest
4) Feature engineering (pack-years feature from smoking data)
5) Feature selection (Best subset selection, recursive elimination)
6) Bayesian optimisation hyper-parameter tuning






