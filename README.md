# Disaster response project
A model that classifies messages sent out during natural disasters.

## Project summary
The purpose of this project is to train a machine learning model that can classify messages sent during natural disasters as belonging to one or more of 36 categories. This kind of model can help emergency services find relevant messages amidst thousands of pieces of information sent out during a natural disaster. Besides training a model, this project also creates a front end (to be deployed locally) to test the model and classify messages interactively.

## What's inside
The project has three main folders and a script, `run.py`.
### Data
This folder contains two .csv files containing the raw data used for the model. The file `disaster_categories.csv` contains, for each message, which categories it belongs to. `disaster_messages` contains the content of each message, the original (if not in English), and what kind of communication it was (direct, through news, or social media). 

Additionally, the `data` folder contains a Python script, `process_data.py`, that merges and cleans these datasets to be usable for machine learning and saves the new dataset as a SQL database in this same folder.

### Models
The `models` folder initially only contains a Python script, `train_classifier.py`, which will load in the data and train a machine learning model on it. It will prepare the data for natural language processing and fit a model, tune it using grid search, report accuracy, and save the final model in a Pickle file called `classifier.pkl` in this folder.

### Templates
This folder contains two HTML files that will render the front-end of the Flask app. 

### run.py
This script will execute the whole project: ingesting and cleaning data, training and tuning the model, and creating the front-end Flask app. 

## How to run
Run the following commands in the project's root directory:
  ### To create the database (ETL pipeline):
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  ### To build the model (ML pipeline):
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
  ### To deploy the Flask app to a local server
      `python run.py`
  ### To access the app
      Go to http://0.0.0.0:3001/
