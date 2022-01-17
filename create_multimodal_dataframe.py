import numpy as np
import pandas as pd
import pickle
import argparse

# Argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--audio', required=True,
	help = "path to the audio dataframe")
parser.add_argument('-v', '--video', required=True,
	help = "path to the video dataframe")
args = vars(parser.parse_args())

def combine_dataframes(audio, video):
    # Reading in the unimodal dataframes
    df_audio = pd.read_pickle(audio)
    df_video = pd.read_pickle(video)

    # Combining unimodal dataframes
    df_mm = df_audio.join(df_video, how='inner', rsuffix='v')

    # True if everything went well
    print("Verifying combination...")
    print(f"Emotion lables are identical: {all(df_mm['LABEL'] == df_mm['LABELv'])}")
    print(f"Activation lables are identical: {all(df_mm['ACTIVATION'] == df_mm['ACTIVATIONv'])}")
    print(f"Valence lables are identical: {all(df_mm['VALENCE'] == df_mm['VALENCEv'])}")

    # Dropping repeating columns
    df_mm.drop(['LABELv', 'ACTIVATIONv', 'VALENCEv'], axis=1, inplace=True)

    # Renaming columns with acoustic and visual features accordingly
    df_mm.rename(columns={'FEATURES': 'AUDIO', 'FEATURESv': 'VIDEO'}, inplace=True)

    return df_mm

if __name__ == '__main__':
    # Running
    print("Running...")
    df_multimodal = combine_dataframes(args['audio'], args['video'])

    # Storing the new audio-visual dataframe
    df_name = 'df_multimodal.pkl'
    df_multimodal.to_pickle(df_name)
    print(f'DONE running! (Saved as: {df_name})')