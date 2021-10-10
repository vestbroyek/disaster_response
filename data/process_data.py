import sys


# importing relevant libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# loading in data
def load_data(messages_filepath, categories_filepath):
    """Ingests two csvs (messages and categories) and merges them on their ID columns
    
    Parameters
    -----------
    messages_filepath : str
            The filepath to the messages csv
    categories_filepath : str
            The filepath to the categories csv
            
    Returns
    -----------
    df
            The merged pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
# merging dfs on id column
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    return df

# cleaning data
def clean_data(df):
    """Cleans the dataframe: creates clear category columns, fixes headers, removes duplicates
    
    Parameters
    -----------
    df : a pandas DataFrame
            The dataframe to be cleaned
            
    Returns
    -----------
    df : a pandas DataFrame
            The cleaned dataframe
    """
    # splitting categories into separate columns
    # first create a separate dataframe of the 36 categories
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # rename columns: values like "request-1" need to become col names like
    # "request"
    columns = []
    
    for string in categories[:1].values.tolist()[0]:
        string = string.replace('-', '').replace('0', '').replace('1', '')
        columns.append(string)
    
    # rename the columns of the "categories" df
    categories.columns = columns
    
    # convert values to 0 or 1 (from e.g. "request-0" to just int 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from str to itn
        categories[column] = categories[column].astype(int)
    
    # dropping rows where 'related' = 2 (which makes no sense)
    categories = categories[categories['related'] != 2]
        
    # replacing categories column in df with the cleaned category columns
    # first drop the old column
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new categories df
    df = pd.concat([df, categories], axis = 1)
    
    # remove duplicates (170 of 26,386)
    df.drop_duplicates(inplace = True)
    
    return df
    
def save_data(df, database_filename):
    """Saves the clean dataframe to a SQL database
    
    Parameters
    -----------
    df : a pandas DataFrame
            The dataframe to be cleaned
    database_filename : str
            A filename for the SQL database to be created
            
    Returns
    -----------
    None
    """
    # engine = create_engine(database_filename)
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index = False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
