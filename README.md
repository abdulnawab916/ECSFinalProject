# ECS 171 - Machine Learning -  Heart Disease Prediction Model
Final project repo for ECS 171 Machine Learning Class.

## Exploring Data Analysis
``EDA.ipynb`` - Compares bagged decision tree, and logistic regression

## HOW TO RUN THE APP:
1) Open the ``flask`` folder in terminal.
2) Run ``python app.py`` on the command line
3) Open the URL (localhost) on your browser.
- It should look something like this: ``http://127.0.0.1:5000/``

## MAKING CHANGES TO DATASET:
- If you're changing the model, Make sure to rerun ``python models.py`` in the ``Exploratory Data Analysis`` folder to create a new "bagged_decision_tree_model.pkl" file.

**ALL NEW INFORMATION NOW IN ``EDA.ipynb``**

### UPDATE LOG

**V0.1** 
- got the starter app set up with a "basicheart.csv", which is a reduced dataset.

**V1.0** 
- Refined starter app, uses the full dataset "heart.csv"
- Only uses logistic model for prediction
- Organized the GitHub Repo

**V1.1** 
- Made starter app better
- Did more EDA

**V2** 
- FIXED OUR DATASET
    - inverted target with 0 to 1
- FIXED EDA
- FIXED EVERYTHING