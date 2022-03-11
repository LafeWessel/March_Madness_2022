
import pandas as pd

def calculate_score(data: pd.DataFrame):
    """Calculate the score for a given set of 10 choices, index 0 is top prediction, 9 lowest"""
    # assumes that data has 10 entries, 0-9, 0 for top, 9 for bottom
    assert data.shape[0] >= 10

    # assumes that `Cinderella` and `Number of Tournament Wins` are in columns
    assert "Cinderella" in data.columns
    assert "Number of Tournament Wins" in data.columns

    # reset index
    df = data.copy()
    df.reset_index(drop=True, inplace=True)

    score = 0
    # calculate the score
    for i in range(0,10):

        # calculate for top 10
        score += (10-i) * df.at[i,'Number of Tournament Wins']
        
        # account for cinderella teams
        if df.at[i,'Cinderella'] == 1:
            score += 5
    # end for
    return score
# end def

if __name__ == "__main__":
    print("Running mm_utils.py")

    print("Testing calculate_score")

    df = pd.DataFrame(columns=['Cinderella', 'Number of Tournament Wins'])
    df = df.append({"Cinderella":0,"Number of Tournament Wins":2}, ignore_index=True) #10 score += 20
    df = df.append({"Cinderella":1,"Number of Tournament Wins":1}, ignore_index=True) #9 score += 14
    df = df.append({"Cinderella":0,"Number of Tournament Wins":1}, ignore_index=True) #8 score += 8
    df = df.append({"Cinderella":1,"Number of Tournament Wins":4}, ignore_index=True) #7 score += 33
    df = df.append({"Cinderella":0,"Number of Tournament Wins":5}, ignore_index=True) #6 score += 30
    df = df.append({"Cinderella":1,"Number of Tournament Wins":1}, ignore_index=True) #5 score += 10
    df = df.append({"Cinderella":0,"Number of Tournament Wins":0}, ignore_index=True) #4 score += 0
    df = df.append({"Cinderella":1,"Number of Tournament Wins":0}, ignore_index=True) #3 score += 5
    df = df.append({"Cinderella":0,"Number of Tournament Wins":2}, ignore_index=True) #2 score += 4
    df = df.append({"Cinderella":1,"Number of Tournament Wins":1}, ignore_index=True) #1 score += 6
    
    # total score = 20 + 14 + 8 + 33 + 30 + 10 + 0 + 5 + 4 + 6 = 130
    expected = 130
    score = calculate_score(df)
    print(f"calculated score {score}, expected {expected}")
    assert score == expected
