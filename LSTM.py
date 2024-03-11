import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

def load_and_preprocess_data(file_path):
    # Load the spreadsheet
    df = pd.read_excel(file_path)
    
    # Assuming the first four columns are features and the fifth column is the target
    features = df.iloc[:, :4]  # First four columns as features
    target = df.iloc[:, 4]  # Fifth column as target
    
    return features, target

# Initialize model creation function
def create_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


training_files = ['C:\\Users\\mtpv1\\Downloads\\Set 1_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 2_Processed.xlsx', 
                  'C:\\Users\\mtpv1\\Downloads\\Set 3_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 4_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 5_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 6_Processed.xlsx', 
                  'C:\\Users\\mtpv1\\Downloads\\Set 7_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 8_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 9_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 10_Processed.xlsx', 
                  'C:\\Users\\mtpv1\\Downloads\\Set 11_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 12_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 13_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 14_Processed.xlsx', 
                  'C:\\Users\\mtpv1\\Downloads\\Set 15_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 16_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 17_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 18_Processed.xlsx', 
                  'C:\\Users\\mtpv1\\Downloads\\Set 19_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 20_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 21_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 22_Processed.xlsx']

testing_files =  ['C:\\Users\\mtpv1\\Downloads\\Set 23_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 24_Processed.xlsx',
                  'C:\\Users\\mtpv1\\Downloads\\Set 25_Processed.xlsx', 'C:\\Users\\mtpv1\\Downloads\\Set 26_Processed.xlsx']

# Initialize the model once, outside the loop
# Assuming all files have the same feature shape, we can set a fixed input shape
model = create_model(input_shape=(4, 1))  # Adjust the shape based on your features

# Load and preprocess training data
for file in training_files:
    features, target = load_and_preprocess_data(file)
    # Train the model on the current file's data
    model.fit(features, target, epochs=10, batch_size=5)  # Adjust epochs & batch_size as needed

# Loop through each testing file to evaluate the model
for file in testing_files:
    features, target = load_and_preprocess_data(file)
    # Evaluate the model on the current file's data
    loss = model.evaluate(features, target)
    print(f"Evaluation on {file.split('\\')[-1]}: Loss = {loss}")

# Save the model
model.save('195_Epidemic_Model.h5')  # Saves the model to a HDF5 file
