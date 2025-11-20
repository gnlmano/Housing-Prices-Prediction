# Course: Data Science for Decision Making

## Module: Computational Machine Learning I - Assignment I (Instructions for assignment)

### Linear Regression:

Regression project: Real Estate Assessment Evaluation

Home valuation is key in real estate industry, and also the basis for mortgages in credit sector. Here we have to predict the estimated value of a property.

Data (RegressionSupervisedTrain.csv) consist of a list of features plus the resulting "parcelvalue", described in "Casedatadictionary.xlsx" file. Each row corresponds to a particular home valuation, and "transactiondate" is the date when the property was effectively sold. Properties are defined by "lotid", but be aware that one property can be sold more than once (it's not the usual case). Also notice that some features are sometimes missing, your model has to deal with it.

Note that you should not use "totaltaxvalue", "buildvalue" or "landvalue", because they are closely correlated with the final value to predict. There is a further member of the training set predictors which is not available in the test set and therefore needs removing. Using this data build a predictive model for "parcelvalue"

In your analysis, use the RMSE (Root Mean Squared Error) criterion for choosing any hyperparameters. Try a first quick implementation, then try to optimize hyperparameters.

For this analysis there is an extra test dataset. Once your code is submitted we will run a competition to see how you score in the test data. Hence, have prepared also the necessary script to compute the MSE estimate on the test data once released.
Bonus: Try an approach to fill NA without removing features or observations, and check improvements.

You can follow those steps in your first implementation:
* Explore and understand the dataset. Report missing data
* Remove columns 'totaltaxvalue', 'buildvalue' or 'landvalue' from the training and testing set and also 'mypointer' from the training set
* As a simplified initial version, get rid of missing data by:
    - Removing features that have more than 40% of missing data in the training set (remember anything you remove from the training set must be removed form the testing set!) (HINT: data.dropna(axis=1, thresh=round(mypercentagevalid*len(data.index)) - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
    - After that, removing observations that have missing data
* Create dummy variables for relevant categorical features (EXTENDED PROJECT ONLY)
* Build your model and test it on the same input data
* Assess expected accuracy using cross-validation
* Report which variable impacts more on results
* Prepare the code to run on a new input file and be able to report accuracy, following same preparation steps (missing data, dummies, etc)

Note that you should not use *totaltaxvalue*, *buildvalue *or *landvalue*, because they are closely correlated with the final value to predict.

You may want to iterate to refine some of these steps once you get performance results in step 5.

# Extension to deployment code:
Housing Price Prediction — End-to-End ML Pipeline + API Deployment

This project builds an end-to-end machine learning system to predict housing prices using tabular data (parcel-level features, geographic encoding, property characteristics, etc.).

Work done:
- Data cleaning & preprocessing pipeline
- Feature engineering
- Geospatial binning + target encoding
- Model training with cross-validation & hyperparameter tuning
- Fully reproducible scikit-learn pipeline
- Dockerized FastAPI prediction service
- Automated tests (pytest)

## To run locally:
1. Create environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python src/training/train_model.py
