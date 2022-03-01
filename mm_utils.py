
import pandas as pd

def calculate_score(data: pd.DataFrame, )
    """Calculate the score for a given set of 10 choices, index 0 is top prediction, 9 lowest"""
    # assumes that data has 10 entries, 0-9, 0 for top, 9 for bottom
    assert data.shape[0] == 10

    # assumes that `Cinderella` and `Number of Tournament Wins` are in columns
    assert "Cinderella" in data.columns
    assert "Number of Tournament Wins" in data.columns

    score = 0
    # calculate the score
    for i in range(0,10):

        # calculate for top 10
        score += (10-i) * data.at[i,'Number of Tournament Wins']
        
        # account for cinderella teams
        if data.at[i,'Cinderella'] == 1:
            score += 5
    # end for
    return score




if __name__ == "__main__":
    print("Running mm_utils.py")
