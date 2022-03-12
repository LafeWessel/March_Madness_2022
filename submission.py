import os as os
import pickle as pk
import tensorflow as ts
import pandas as pd
import zipfile
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def predict_wins(model, observation):
    """Make a prediction with the given model and return the number that was predicted"""
    pred = model.predict(observation).values.reshape(1,-1))
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
    try:
        df = pd.read_excel("March Madness 2022.xlsx", engine='openpyxl')
    except:
        df = pd.read_excel("March Madness 2022.xlsx")

    # extract pickled models
    pickle_dir = "pickled_models"
    os.mkdir(pickle_dir)
    zipfile.Zipfile("submission_models.zip",'r').extractall(pickle_dir)

    # read in pickled models
    models = dict()
    for p in glob.glob(os.path.join(pickle_dir,"*.pkl")):
        name = os.path.split(p)
        name = name.replace("_pca.pkl","")
        name = name.replace("_sclr.pkl","")

        models[name] = pk.load(open(p,'r'))
    # end for

    # read in tf model
    tf_model = pk.load(open(os.path.join(pickle_dir,"neural_network.tfmodel"),'r'))

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
    for s in to_scale:
        df[s] = models[s].transform(df[[s]])

    # reduce with PCA
    for k in pca_combinations.keys():
        res = pd.DataFrame(models[k].transform(df[pca_combinations[k][0]]), columns=[f'{k}_{n}' for n in range(pca_combinations[k][1])])
        df = df.join(res)
        df.drop(columns=pca_combinations[k][0], inplace=True)

    # predict with neural network
    if "Number of Tournament Wins" in df.columns:
        df.drop(columns=['Number of Tournament Wins'], inplace=True)
    df['Predicted Wins'] = df.drop(columns=['Team','Region','Conference','Conference Tournament Champion','Made Tournament Previous Year']).apply(lambda r: predict_wins(tf_model, r))
    df.sort_values(['Predicted Wins', 'Cinderella'], ignore_index=True, inplace=True, ascending=False)

    res = df.iloc[:10,:]
    res.sort_values(['Predicted Wins', 'Cinderella'], ascending=True, ignore_index=True, inplace=True)

    # display results
    print(res)
    
