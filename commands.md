# Readme for March Madness

## Process

### Data Cleaning

- Standard Scale (almost) everything
- OHE for Region
- Don't care about conference
- Don't care about Total Points, Opp Pts, Scoring Diff
- Use 3pt % more
- Don't use Conference Tournament Champions -> might be a negative

Look at Quad 1 Wins and compare with Wins against top 25 teams to see how different they are.

#### Small model

- PCA for some smaller $n$ columns
- Handpick feautures -> 3 pt.

### Neural Network Model

- Several (5?) fully-connected layers
- 6 output nodes for number of wins
    - apply softmax function

#### Other?

- Make output 0-10 for what prediction for team should be to get an optimal score. 1-10 are place predictions, 0 for not selected.

### Post Processing

Take predictions of number of wins and select top $n$ to keep as win predictions.
- Use 4-6 for $n$ of Cinderella teams
Give top slots to top predicted teams.

### Evaluate

Apply scoring algorithm to determine how well the model performs.

## Commands

### Start Pip

source ./bin/activate

### Start Jupyter Notebook

jupyter notebook --port 8888 --no-browser --ip 0.0.0.0

