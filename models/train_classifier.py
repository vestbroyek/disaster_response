import sys

# importing relevant libraries
import pandas as pd
import numpy as np
# db
import sqlalchemy as db
# regex
import re
# nlp libraries
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
# sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
# pickle
import pickle

# loading in data
def load_data(database_filepath):
    # database_filepath = './data/DisasterResponse.db'                                  ## Is this correct?
    database_filepath = 'data/DisasterResponse.db'
    name = 'sqlite:///' + database_filepath
    engine = db.create_engine(name)
    conn = engine.connect()
    df = pd.read_sql_table('DisasterResponse', con = engine)
    # feature variable
    X = df['message']
    # target variables - 36 category cols
    Y = df.iloc[:, 4:]
    # category names
    category_names = Y.columns
    
    return X, Y, category_names

# tokenising text
def tokenize(text):
    # initialising lemmatiser
    lemmatizer = WordNetLemmatizer()
    # loading in stop words
    stop_words = stopwords.words('english')
    # normalising case and removing punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    # tokenising text
    tokens = word_tokenize(text)
    # lemmatise
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

# training the model
def build_model():
    # building a pipeline
    model = Pipeline([                                                              # calling this 'model' so it can be fit under __main__
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    # splitting data into train and test
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)  # commenting out for now because train_test is under __main__
    
    # training model on pipeline
    # model = pipeline.fit(X_train, Y_train)                                        # also commenting out because model.fit is under __main__
    
    return model

def evaluate_model(model, X_test, Y_test):
    # make predictions
    Y_pred = model.predict(X_test)
    
    # generate and print precision, recall, f1-score for each output category
    for i in range(36):
        print(classification_report(Y_pred[:, i], np.array(Y_test)[:, i]))
        
    # evaluate predictions
    accuracy = (Y_pred == Y_test).mean()
    print("Overall accuracy: ", accuracy)
    
    # Grid Search
    # hyperparameters
    parameters =  {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 4],
    }

    grid = GridSearchCV(pipeline, param_grid = parameters)
    grid.fit(X_train, Y_train)
    print('Best parameters: ', grid.best_params_)

    final_model = grid.best_params_
    
    return final_model

def save_model(final_model, model_filepath):                                    ## and is this correct?
    filename = 'classifier.pkl'
    model_filepath = 'models/classifier.pkl'
    pickle.dump(final_model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()