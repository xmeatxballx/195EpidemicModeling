import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    # Load the spreadsheet
    df = pd.read_excel(file_path)
    
    # Assuming the first four columns are features and the fifth column is the target
    features = df.iloc[:, :4]  # First four columns as features
    target = df.iloc[:, 4]  # Fifth column as target
    
    return features, target

testing_files =  ['C:\\Users\\mtpv1\\Downloads\\Set 23_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 24_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 25_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 26_Processed.xlsx']


model = load_model('C:\\Users\\mtpv1\\Documents\\GitRepos\\195EpidemicModeling\\195_Epidemic_Model.h5')

#for file in testing_files:
#    features, target = load_and_preprocess_data(file)
#    # Evaluate the model on the current file's data
#    loss = model.evaluate(features, target)
#    print(f"Evaluation on {file.split('\\')[-1]}: Loss = {loss}")

df = pd.read_excel('C:\\Users\\mtpv1\\Downloads\\Data.xlsx')

# Select relevant columns (if necessary)
df = df[['Set 9']]

predictions = model.predict(df)
plt.figure(figsize=(10, 6))
plt.plot(df, label='Actual Cases')
plt.plot(predictions, label='Predicted Cases', alpha=0.7)
plt.title('Model Predictions vs Actual Cases (Washington D.C.)')
plt.xlabel('Weeks since Beginning of Pandemic')
plt.ylabel('Covid Cases per Capita')
plt.legend()
plt.show()