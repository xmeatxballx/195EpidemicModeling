import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Load the data
new_working_directory = '/Users/ya/Downloads/networkSIR/'  #Setting the working directory
os.chdir(new_working_directory)
data = pd.read_csv('NSW_data.csv')

dates = pd.to_datetime(data['Date'])
S = data['S'].str.replace(',', '').astype(float).values 
I = data['I'].str.replace(',', '').astype(float).values
R = data['R'].str.replace(',', '').astype(float).values

beta_values = data['beta']
gamma_values = data['gamma']

# Plotting
plt.figure(figsize=(10, 6))

# Plot each category as a line
plt.plot(dates, S, label='Susceptible', color='blue')
plt.plot(dates, I, label='Infected', color='red')
plt.plot(dates, R, label='Recovered', color='green')

plt.xlabel('Date')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Integrate the model for each set of beta and gamma values
for i in range(len(data)):
    beta = beta_values[i]
    gamma = gamma_values[i]
    
    # Initial conditions
    S0 = data['S'].str.replace(',', '').astype(float).values 
    I0 = data['I'].str.replace(',', '').astype(float).values
    R0 = data['R'].str.replace(',', '').astype(float).values
    

    # Time points
    t = np.linspace(0, len(dates)-1, len(dates))
    y0 = np.array([S0[i], I0[i], R0[i]])
    
    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    print(f"Simulation {i+1}:")
    print(f"Initial conditions: S0 = {S0[i]}, I0 = {I0[i]}, R0 = {R0[i]}")
    print(f"Parameters: beta = {beta}, gamma = {gamma}")

   
    plt.figure(figsize=(10, 6))
    plt.plot(dates, solution[:, 0], label='Susceptible', color='blue')
    plt.plot(dates, solution[:, 1], label='Infected', color='red')
    plt.plot(dates, solution[:, 2], label='Recovered', color='green')
    plt.xlabel('Date')
    plt.ylabel('Number of individuals')
    plt.title(f'SIR Model (beta={beta}, gamma={gamma})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()