import pandas as pd




def detect_column_types(df):
 
    numerical_columns=df.select_dtypes(include="number").columns.tolist()
    categorical_columns=df.select_dtypes(include="object").columns.tolist()
    for entry in numerical_columns:
        print(entry)
    print("=========================")
    for entry in categorical_columns:
        print(entry)


def load_dataset_excel(file_path):
    df=pd.read_excel(file_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Columns:", df.columns.tolist())
    return df

def load_dataset_csv(file_path):
    df=pd.read_csv(file_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Columns:", df.columns.tolist())
    return df

def data_summary(df):
    print("Dataset Info:")
    print(df.info())
    print("\n Missing Values:")
    print(df.isna().sum())
    print("\n Descriptive Statistics")
    print(df.describe(include="all"))

def convert_to_datetime(df,column):
    df[column]=pd.to_datetime(df[column])
    return df

def unique_value_count_summary(df,columns):
    print(f"Value counts for {columns}")
    print(df[columns].value_counts())

def rename_columns(df, columns_dict):
    return df.rename(columns=columns_dict)