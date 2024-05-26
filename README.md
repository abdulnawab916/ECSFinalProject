# ECS 171 - Machine Learning -  Heart Disease Prediction Model
Final project repo for ECS 171 Machine Learning Class.

**V0.1** - got the starter app set up with a "basicheart.csv", which is a reduced dataset.

**V1.0** 
- Refined starter app, uses the full dataset "heart.csv"
- Only uses logistic model for prediction
- Organized the GitHub Repo

**V1.1** 
- Made starter app better
- Did more EDA

## Exploring Data Analysis
``EDA_V1.py`` - Compares Bagged Decision Tree vs Random Forest
``EDA_V2.py`` - Compares Bagged Decision Tree vs Logistic Regression

## HOW TO RUN THE APP:
1) Open the ``flask`` folder in terminal.
2) Run ``python app.py`` on the command line
3) Open the URL (localhost) on your browser.
- It should look something like this: ``http://127.0.0.1:5000/``

## MAKING CHANGES TO DATASET:
- If you're changing the dataset, Make sure to rerun ``python logisticmodel.py`` in the ``flask`` folder to create a new "logistic_regression_model.pkl" file.

## NOTES
- We need to do more EDA.
    - The report should include our EDA where we discussed about the Logistic Regression and the Bagged Decision Tree.
    - Basically, we found that the Logistic Regression was more inaccurate than the Bagged Decision Tree, which shows our data is more non linear. The Bagged Decision Tree had an accuracy of 98.54% compared to Logistic Regression accuracy, where it was 79.51%.
- Color theory -- We need to make the UI of the landing page for the website better, so it's more related to our "Heart Disease" theme.
- Make the dropdowns better. We need to not use the default HTML format? 


- We also found out that Bagged Decision Trees vs Random Forest have the same accuracy:
Bagged Decision Tree Accuracy: 0.9853658536585366
Random Forest Accuracy: 0.9853658536585366
Bagged Decision Tree Cross-Validation Accuracy: 0.9970731707317073
Random Forest Cross-Validation Accuracy: 0.9970731707317073