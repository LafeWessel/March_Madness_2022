import os as os
import pickle as pk

import numpy as np
import tensorflow as ts
import pandas as pd
import glob as glob
import zipfile
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from warnings import filterwarnings
filterwarnings('ignore')


def predict_wins(model, observation):
    """Make a prediction with the given model and return the number that was predicted"""
    pred = model.predict(observation.values.reshape(1, -1))
    num = pred[0].argmax()
    return num


if __name__ == "__main__":
    # Must:
    #   read in new data
    #   read in scaling models
    #   read in neural network
    #   apply scaling (if necessary) to data
    #   predict with neural network
    #   create output

    # read in new data
    print("Attempting to read data from March Madness 2022.xlsx")
    try:
        df = pd.read_excel("March Madness 2022.xlsx", engine='openpyxl')
    except:
        df = pd.read_excel("March Madness 2022.xlsx")

    # extract pickled models
    print("Extracting pickled models from submission_models.zip")
    pickle_dir = "pickled_models"
    if pickle_dir not in os.listdir():
        os.mkdir(pickle_dir)
    zipfile.ZipFile("pickled_models.zip", 'r').extractall(pickle_dir)

    # read in pickled models
    print("Loading pickled models")
    models = dict()
    for p in glob.glob(os.path.join(pickle_dir, "*.pkl")):
        print(f"\tReading in {p}")
        name = os.path.split(p)[-1]
        name = name.replace("_pca.pkl", "")
        name = name.replace("_sclr.pkl", "")

        models[name] = pk.load(open(p, 'rb'))
    # end for

    # unzip tf model
    print("Extracting neural network from neural_network.zip")
    zipfile.ZipFile("neural_network.zip", 'r').extractall(".")

    # read in tf model
    print("Loading model from ./neural_network/")
    tf_model = ts.keras.models.load_model("neural_network")

    # columns to scale
    to_scale=['Game Count', 'Wins',
       'Losses', '3-Pointers Made', '3-Pointers Attempted',
       '3-Point Percentage', 'Free Throws Made', 'Free Throws Attempted',
       'Free Throw Percentage', 'Rebounds', "Opponent's Rebounds",
       'Rebound Differential', 'Offensive Rebounds', 'Assists', 'Turnovers',
       'Assist to Turnover Ratio', 'ESPN Strength of Schedule',
       'Wins Against Top 25 RPI Teams', 'Losses Against Top 25 RPI Teams',
       'Total Points', 'Average PPG', 'Total Opp Points', 'Average Opp PPG',
       'Total Scoring Differential', 'Scoring Differential Per Game',
       'Quad 1 Wins', 'Quad 1 Losses']

    pca_combinations={
        "3pt":(['3-Pointers Made','3-Pointers Attempted','3-Point Percentage'],1),
        "Free Throw-Rebound":(['Free Throws Made','Free Throws Attempted','Free Throw Percentage','Rebounds',"Opponent's Rebounds","Rebound Differential","Offensive Rebounds"],3),
        "Region":(['Region_East','Region_Midwest','Region_South','Region_West'],1),
        "PPG":(['Total Points', 'Average PPG', 'Total Opp Points', 'Average Opp PPG'],2),
        "Scoring Differential":(['Total Scoring Differential','Scoring Differential Per Game'],1),
        "Assist":(['Assists', 'Assist to Turnover Ratio', 'Turnovers'],1),
        "Schedule":(['ESPN Strength of Schedule', 'Wins vs Top Teams','Losses vs Top Teams'],1),
        "Win-Loss":(['Game Count','Wins','Losses'], 1)
    }

    # scale data
    print("Scaling columns")
    for s in [s for s in to_scale if s in df.columns]:
        print(f"\tScaling {s}")
        df[s] = models[s].transform(df[s].values.reshape(-1, 1))

    # add dummy columns for Region
    print("Creating dummy columns for Region")
    df = df.join(pd.get_dummies(df['Region'], prefix="Region"))

    # move Quad 1 Wins & Wins Against Top 25 Teams to Wins vs Top Teams
    df['Wins vs Top Teams'] = np.NaN
    if "Quad 1 Wins" in df.columns:
        df['Wins vs Top Teams'].fillna(df['Quad 1 Wins'], inplace=True)
        df.drop(columns=['Quad 1 Wins'], inplace=True)
    if "Wins Against Top 25 Teams" in df.columns:
        df['Wins vs Top Teams'].fillna(df['Wins Against Top 25 Teams'], inplace=True)
        df.drop(columns=['Wins Against Top 25 Teams'], inplace=True)

    # move Quad 1 Losses & Losses Against Top 25 Teams to Losses vs Top Teams
    df['Losses vs Top Teams'] = np.NaN
    if "Quad 1 Losses" in df.columns:
        df['Losses vs Top Teams'].fillna(df['Quad 1 Losses'], inplace=True)
        df.drop(columns=['Quad 1 Losses'], inplace=True)
    if "Losses Against Top 25 Teams" in df.columns:
        df['Losses vs Top Teams'].fillna(df['Losses Against Top 25 Teams'], inplace=True)
        df.drop(columns=['Losses Against Top 25 Teams'], inplace=True)

    # reduce with PCA
    print("Reducing feature count with PCA")
    for k in pca_combinations.keys():
        print(f"Reducing {pca_combinations[k][0]} to {pca_combinations[k][1]} columns")
        res = pd.DataFrame(models[k].transform(df[pca_combinations[k][0]]), columns=[f'{k}_{n}' for n in range(pca_combinations[k][1])])
        df = df.join(res)
        df.drop(columns=pca_combinations[k][0], inplace=True)

    # predict with neural network
    if "Number of Tournament Wins" in df.columns:
        df.drop(columns=['Number of Tournament Wins'], inplace=True)

    print("Making predictions")
    df['Predicted Wins'] = df.drop(columns=['Team','Region','Conference','Conference Tournament Champion','Made Tournament Previous Year']).apply(lambda r: predict_wins(tf_model, r), axis=1)
    df.sort_values(['Predicted Wins', 'Cinderella'], ignore_index=True, inplace=True, ascending=False)

    res = df.iloc[:10, :]
    res.sort_values(['Predicted Wins', 'Cinderella'],ignore_index=True, inplace=True, ascending=True)

    # select 3 Cinderellas to occupy bottom spots
    cind = df.loc[~df.Team.isin(res.Team)]
    cind = cind[cind['Cinderella'] == 1].iloc[:3, :]
    res.iloc[:3, :] = cind

    # display results
    print("\n\n-----Predictions-----")
    for i in res.index:
        print(f"\t{10-i}: {res.at[9-i,'Team']}")

    print("\n\nCleaning up decompressed files")

    print("\tRemoving pickled_models/")
    shutil.rmtree(pickle_dir)

    print("\tRemoving neural_network/")
    shutil.rmtree("neural_network")
    
