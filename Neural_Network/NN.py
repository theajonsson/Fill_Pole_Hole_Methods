"""
File:       NN.py
Purpose:    Neural network (NN) architecture Multi Layered Perceptron based on Soriot et al. (2022)
            (https://doi.org/10.1029/2022EA002542)
            Training NN (using PyTorch )on a dataset of temperature brightness (TB) [K] on different microwave channels 
            and sea ice thickness (SIT) [m] for corresponding time period
            PLOTS
            RUN MODEL ON A DIFFERENT DATASET

Function:   train_model, NN_automated

Other:      Created by Thea Jonsson 2025-08-19
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

mpl.rcParams['agg.path.chunksize'] = 10000


# NN model using MLP architecture
# Input: brightness temperature (TB) [K] -> 5 different frequencies and polarization (V19, H19, V22, V37, H37)
# Output: sea ice thickness (SIT) [m]
class Model(nn.Module):
    def __init__(self, in_features=5, n_hidden=4, n_outputs=1):
        print(f"From NN.py n_hidden = {n_hidden}")
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(in_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        #self.hidden3 = nn.Linear(n_hidden, n_hidden)
        #self.hidden4 = nn.Linear(n_hidden, n_hidden)
        #self.hidden5 = nn.Linear(n_hidden, n_hidden)
        self.activation = nn.Tanh()                      
        self.output = nn.Linear(n_hidden, n_outputs)    

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        #x = self.hidden3(x)
        #x = self.activation(x)
        #x = self.hidden4(x)
        #x = self.activation(x)
        #x = self.hidden5(x)
        #x = self.activation(x)
        x = self.output(x)          
        return x         



"""
Function:   train_model
Purpose:    Trains a NN to predict SIT from a set of TB, training data is feed forward trough the network,
            uses the loss between predicted and true values to backpropogate and update parameters for next epoch,
            Optinal to save the trained model and scaler, also see the learned input weights in a .csv file 

Input:      data (.csv file): it contains TB values for V19, H19, V22, V37, H37 and SIT values and corresponding x/y coordinates 
            seed: random seed to make results reproducible
            test_size: fraction of training data to use for testing (0.3 means 30% of training data is used for testing)
            random_state: random split seed
            lr: learning rate for the optimizer
            epochs: number of runs to iterate over the whole training dataset
Return:     N/A
            model_save (if return_model is True): model and scaler
"""
def train_model(data, 
                seed=100, test_size=0.3, random_state=42, lr=0.01,      
                epochs=200,
                n_hidden=1, return_model=False):    

    torch.manual_seed(seed)       # Random manual seed for randomization

    X = data[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values        # input to model (TB)
    y = data[["SIT"]].values                                                   # what model should predict (SIT)

    # Scale input feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train) 
    y_test = torch.FloatTensor(y_test)

    # Criterion of model to measure the error
    model = Model(n_hidden=n_hidden)              # Instance for model 
    criterion = nn.MSELoss()     # Mean Squared Error (MSE) Loss - cont. numerical value, calc. squared diff. between predicted and target values
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   

    # Train the model
    losses = []
    for epoch in range(epochs):

        y_pred = model.forward(X_train)   # Prediction results

        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy())

        if epoch % 50 == 0:
            print(f"Epoch {epoch} and loss: {loss}")
        
        # Back propogation (to fine tune the weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Training is done")

    # Plot of loss/error against each epoch
    if False:
        plt.plot(range(epochs), losses)
        plt.ylabel("loss/error")
        plt.xlabel("Epoch")
        plt.savefig("/Users/theajonsson/Desktop/Loss.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Save learned input weights 
    if False:
        input_weights = model.hidden1.weight.detach().numpy()
        channel_names = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]
        weights_df = pd.DataFrame(input_weights.T, index=channel_names, columns=[f"Hidden_{i+1}" for i in range(model.hidden1.out_features)])
        weights_df.to_csv("/Users/theajonsson/Desktop/input_channel_weights.csv", index=True)

    # Save model and scaler
    model_save = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler
    }
    #torch.save(model_save, str(Path(__file__).resolve().parent/"Results/NN/NN_Model.pth"))
    #torch.save(model_save, "/Users/theajonsson/Desktop/NN_Model.pth")

    if return_model:
        return model_save
    


    """ ==========================================================================================
            3 different type of plots to check for different things to consider
    ========================================================================================== """
    # 3 plots (divided in channels: 19,22,37): TB (y-axis) vs predicted test SIT (x-axis), with fitted line, R^2 score, RMSE value
    if False:
        with torch.no_grad():
            y_pred = (np.array(model.forward(X_test))).flatten()

        X_test_TB = scaler.inverse_transform(X_test)

        channel_groups = [
            ["TB_V19", "TB_H19"],   
            ["TB_V22"],             
            ["TB_V37", "TB_H37"]    
        ]
        channel_order = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]
        channel_indices = {ch: i for i, ch in enumerate(channel_order)}

        # Making the plot for each groups
        for group in channel_groups:
            fig, axs = plt.subplots(1, len(group), figsize=(7 * len(group), 6), constrained_layout=True)

            if len(group) == 1:
                axs = [axs] 

            for ax, channel in zip(axs, group):
                idx = channel_indices[channel]
                TB_channel = X_test_TB[:, idx]

                slope, intercept, r_value, p_value, std_err = linregress(y_pred, TB_channel)
                r_squared = r_value ** 2
                rmse = mean_squared_error(y_pred, TB_channel, squared=False)

                hb = ax.hexbin(y_pred, TB_channel, gridsize=50, mincnt=6, label=f"$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}")
                ax.plot(y_pred, intercept + slope * y_pred, color="red",
                        label=f"Fitted line")

                ax.set_xlabel("Predicted test SIT [m]")
                ax.set_ylabel(f"{channel} [K]")
                ax.set_title(f"{channel} vs Predicted SIT")
                ax.legend()
                ax.grid(True)

            dir = "/Users/theajonsson/Desktop/"
            name = f"TB_predSIT_{group}.png"
            plt.savefig(dir+name, dpi=300, bbox_inches="tight")
            plt.show()

    # 3 plots (divided in channels: 19,22,37): predicted test SIT (y-axis) vs true test SIT (x-axis), with fitted line, R^2 score, RMSE value
    # Important
    if False:
        channel_groups = [
            ["TB_V19", "TB_H19"],   
            ["TB_V22"],             
            ["TB_V37", "TB_H37"]    
        ]
        channel_order = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]
        channel_indices = {ch: i for i, ch in enumerate(channel_order)}

        y_test_np = y_test.flatten()

        # Making the plot for each groups
        for group in channel_groups:
            fig, axs = plt.subplots(1, len(group), figsize=(7 * len(group), 6))

            if len(group) == 1:
                axs = [axs]

            for ax, channel in zip(axs, group):
                idx = channel_indices[channel]

                X_test_single = X_test.clone()
                X_test_single[:, :] = X_test[:, idx].unsqueeze(1).repeat(1, X_test.shape[1])
                X_test_single[:, idx] = X_test[:, idx]

                with torch.no_grad():
                    y_pred_test = model.forward(X_test_single)
                y_pred_np = y_pred_test.flatten() 
            
                slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
                r_squared = r_value**2
                rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)

                ax.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6, label=f"R-squared: {r_squared:.3f} \nRMSE={rmse:.3f}")      
                ax.plot(y_test_np, intercept + slope * y_test_np, color="red", label=f"Fitted line")

                ax.set_xlabel("True test SIT [m]")
                ax.set_ylabel("Predicted test SIT [m]")
                ax.legend()
                ax.grid(True)
                ax.set_title(f"{channel} channel")
                ax.set_ylim(0, 5)
                ax.set_xlim(0, 5)
        
            plt.tight_layout()
            dir = "/Users/theajonsson/Desktop/"
            name = f"TrueSIT_predSIT_{group}.png"
            plt.savefig(dir+name, dpi=300, bbox_inches="tight")
            plt.close()

    # Plot (scatter or hexbin) of predicted SIT (y-axis) against true SIT (x-axis), with fitted and optimal line, R^2 score, RMSE value
    # Plot histogram of predicted SIT and true SIT
    if True:
        with torch.no_grad():
            y_pred_test = model(X_test)

        y_test_np = y_test.numpy().flatten()        # y_test 30%:an av SIT
        y_pred_np = y_pred_test.numpy().flatten()   # y_pred 30%:an av X_test

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
        r_squared = r_value**2

        # RMSE calculation
        rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)

        bias = np.mean(y_pred_np - y_test_np)
        
        # Scatter plot
        if False:
            plt.scatter(y_test_np, y_pred_np, alpha=0.6)
            plt.plot(y_test_np, intercept + slope * y_test_np, color='red', label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
            plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
            plt.xlabel("True SIT values [m]")
            plt.ylabel("Predicted SIT values [m]")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 5)
            plt.xlim(0, 5)
            plt.show()
        
        # Hexbin plot
        if True:
            plt.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)
            line_fit, = plt.plot(y_test_np, intercept + slope * y_test_np, color="red", label=f"Fitted line")
            line_opt, = plt.plot([0,6],[0,6], color="black", linestyle="--", label="Optimal line")
            stats_text = f"R$^2$={r_squared:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}, Slope={slope:.3f}"
            custom_entry = Line2D([], [], linestyle='none', marker=None, label=stats_text)
            
            plt.legend(handles=[custom_entry, line_fit, line_opt], loc="lower center", bbox_to_anchor=(0.5, -0.35))
            plt.xlabel("True SIT [m]")
            plt.ylabel("Predicted SIT [m]")
            plt.grid(True)
            plt.ylim(0, 6)
            plt.xlim(0, 6)
            plt.savefig("/Users/theajonsson/Desktop/PredSIT_TrueSIT_hexbin.png", dpi=300, bbox_inches="tight")
            plt.show()

        if True: 
            plt.hist(y_test_np, bins=100, color="blue", alpha=0.5 ,edgecolor="black", zorder=3, label="True SIT", range=(y_test_np.min(), y_test_np.max()))
            plt.hist(y_pred_np, bins=100, color="red", edgecolor="black", zorder=1, label="Predicted SIT", range=(y_test_np.min(), y_test_np.max()))
            plt.xlabel("Sea ice thickness [m]")
            plt.ylabel("Frequency")
            plt.legend()
            plt.xlim(0,10)
            plt.savefig("/Users/theajonsson/Desktop/PredSIT_TrueSIT__hist.png", dpi=300, bbox_inches="tight")
            plt.show()





"""
Function:   NN_automated
Purpose:    Trains a NN to predict SIT from a set of TB, training data is feed forward trough the network,
            uses the loss between predicted and true values to backpropogate and update parameters for next epoch,
            Optinal to save the trained model and scaler, also see the learned input weights in a .csv file 

Input:      n: number of hidden nodes
Return:     model: contains model and scaler
"""
def NN_automated(n):
    try:
        data=pd.read_csv("/Users/theajonsson/Desktop/2006_2007_50km/TD/TD_2006_2007.csv")
        #data = pd.read_csv("/Users/theajonsson/Desktop/2006_2007_80km/TD/TD_2006_2007.csv")
    except FileNotFoundError:
        print("Error")
    model = train_model(data, epochs=1000, lr=0.01, n_hidden=n, return_model=True)

    return model


# -------------------- MAIN --------------------       
# Train NN on training dataset
if True:
    start_time = time.time()

    try:
        #data = pd.read_csv(str(Path(__file__).resolve().parent/"Results/TrainingData/TD_2006_2007.csv"))
        data = pd.read_csv("/Users/theajonsson/Desktop/2006_2007_80km/TD/TD_2006_2007.csv")
    except FileNotFoundError:
        print("Error when reading .csv file")
    train_model(data, epochs=1000, lr=0.01, n_hidden=4)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

























""" ========================================================================================== """
# Test the model on a different data set
if False:
    from training_data_SSMIS import synthetic_tracks
    import os

    model = Model()
    NN_model = torch.load("/Users/theajonsson/Desktop/FillHole_test/1day/SSMIS_1day_model.pth")
    model.load_state_dict(NN_model["model_state_dict"])
    scaler = NN_model["scaler"]

    filename = "BTRin20060313000000424SSF1601GL.nc"
    TestData = synthetic_tracks(filename)
    Test_TB = TestData[["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"]].values 
    TB_xy =  TestData[["X_SIT","Y_SIT"]].values 

    Test_TB = scaler.fit_transform(Test_TB)
    Test_TB = torch.FloatTensor(Test_TB)

    with torch.no_grad():
        y_eval = model.forward(Test_TB)
   

    from format_data import format_SIT
    file = "/Volumes/Thea_SSD_1T/Master Thesis/Envisat_SatSwath/2006/03/ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20060331-fv2.0.nc"
    x_SIT, y_SIT, SIT = format_SIT(file)

    from cartoplot import cartoplot
    cartoplot([TB_xy[:,0], x_SIT], [TB_xy[:,1], y_SIT], [np.array(y_eval), SIT],cbar_label="Sea ice thickness [m]")




""" ==========================================================================================
            2 different type of plots to check for different things to consider
    ========================================================================================== """
# Plot of predicted SIT (y-axis) against true SIT (x-axis), with fitted and optimal line, R^2 score, RMSE value
if False:
    X_test = TestData.drop(["SIT", "X_SIT", "Y_SIT"], axis=1).values        # Input to model: TBs for different channels
    y_test = TestData[["SIT"]].values  

    X_test = scaler.fit_transform(X_test)
    X_test = torch.FloatTensor(X_test)

    with torch.no_grad():
            y_pred_test = model(X_test)

    y_test_np = y_test.flatten()        
    y_pred_np = y_pred_test.numpy().flatten()  

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
    r_squared = r_value**2
    
    # RMSE calculation
    rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)
    
    plt.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)
    plt.plot(y_test_np, intercept + slope * y_test_np, color="red", label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
    plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
    plt.xlabel("True SIT [m]")
    plt.ylabel("Predicted SIT [m]")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    #plt.savefig("/Users/theajonsson/Desktop/SSMIS_1day_hexbin.png", dpi=300, bbox_inches="tight")
    plt.show()

# Plot with 2 subfigures: Cartoplot of NN predicted SIT and true SIT with same x/y coordinates
# Plot of predicted SIT (y-axis) against true SIT (x-axis), with fitted and optimal line, R^2 score, RMSE value
if False:
    Test_TB = TestData.drop(["SIT","X_SIT","Y_SIT"], axis=1).values 
    TB_xy =  TestData[["X_SIT","Y_SIT"]].values 
    SIT = TestData[["SIT"]].values

    Test_TB = scaler.fit_transform(Test_TB)
    Test_TB = torch.FloatTensor(Test_TB)

    with torch.no_grad():
        y_eval = model.forward(Test_TB)

    from cartoplot import multi_cartoplot
    multi_cartoplot([TB_xy[:,0]], [TB_xy[:,1]], [y_eval, SIT], 
                    title=["NN predicted Value"," Real SIT values"],cbar_label="Sea ice thickness [m]")
    
    fig= plt.figure(figsize=[10,5])

    y_test_np = SIT.flatten()       
    y_pred_np = y_eval.flatten() 

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(y_test_np, y_pred_np)
    r_squared = r_value**2

    # RMSE calculation
    rmse = mean_squared_error(y_test_np, y_pred_np, squared=False)

    plt.hexbin(y_test_np, y_pred_np, gridsize = 50, mincnt=6)                  
    plt.plot(y_test_np, intercept + slope * y_test_np, color='red', label=f"Fitted line (R-squared: {r_squared:.3f}, RMSE={rmse:.3f})")
    plt.plot([0,5],[0,5], color="m", linestyle="--", label="Optimal line")
    plt.xlabel("True SIT [m]")
    plt.ylabel("Predicted SIT [m]")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    plt.show()
