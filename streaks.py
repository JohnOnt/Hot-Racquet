import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm



def get_player_points(player, matches, points):
    # Get all match numbers
    match_ids1 = matches[(matches['player1'] == player)].match_id
    match_ids2 = matches[(matches['player2'] == player)].match_id

    player_points1 = points[points.match_id.isin(match_ids1)]
    player_points2 = points[points.match_id.isin(match_ids2)]

    return player_points1, player_points2

#--------------------------------------------------------------------
# Streak Distribution Functions
#--------------------------------------------------------------------

def simulated_iid_series(x,nsims=10,kmax = 40):
    # calculate the number of streaks of a given length for an IDD series with P_M = mean of input series
    tmake = []
    tmiss = []
    p = x.mean()
    for i in range(nsims):
        x = stats.bernoulli.rvs(p,size=x.size)
        make,miss = find_streak_distributions(x,kmax)
        tmake.append(make)
        tmiss.append(miss)
    tmiss = np.array(tmiss)
    tmake = np.array(tmake)
    mu_make = np.mean(tmake,axis=0)
    sig_make = np.std(tmake,axis=0)
    mu_miss = np.mean(tmiss,axis=0)
    sig_miss = np.std(tmiss,axis=0)
    return mu_make,sig_make,mu_miss,sig_miss

def test_streak_distribution_hypothesis(counts,mu,sig,null_only=False):
    # takes input from find_streak_distributions and simulated_series
    if np.sum(counts) == 0:
        # Player has no counts of streaks so must return NA values
        return np.nan, np.nan, np.nan

    kmax = np.max(np.nonzero(counts))+1
    chi2 = np.sum((counts[:kmax]-mu[:kmax])**2/sig[:kmax]**2)
    pval = 1-stats.chi2.cdf(chi2,kmax-1)

    return kmax, chi2, pval

def wald_wolfowitz_distribution(x):
    mu = 2*np.sum(x==0)*np.sum(x==1)/x.size + 1
    sig = np.sqrt( 2*np.sum(x==0)*np.sum(x==1)*(2*np.sum(x==0)*np.sum(x==1)-x.size)/(x.size**2*(x.size-1)))
    return mu,sig

def find_streak_distributions(x,run_length):
    # find number of streaks of misses or makes up to a length 'run_length'
    make_counts = np.zeros(run_length)
    miss_counts = np.zeros(run_length)
    count = 0
    for i in range(1,x.size):
        if x[i]!=x[i-1]:
            if x[i-1]==1:
                make_counts[count] += 1
            else:
                miss_counts[count] += 1
            count = 0
        else:
            if count < (run_length - 1):
                count += 1
    # now do last shot
    if x[i]==1:
        make_counts[count] += 1
    else:
        miss_counts[count] += 1  
    return make_counts,miss_counts

#--------------------------------------------------------------------
# Convert Points into Streaks (Different definition of streak)
#--------------------------------------------------------------------

def streakify_points(points1, points2):
    outcomes = []

    for i in range(np.shape(points1)[0]):
        whosPoint = points1.iloc[i].PointWinner
        if whosPoint == 1:
            outcomes.append(1)
        else:
            outcomes.append(0)

    for i in range(np.shape(points2)[0]):
        whosPoint = points2.iloc[i].PointWinner
        if whosPoint == 2:
            outcomes.append(1)
        else:
            outcomes.append(0)
        
    return np.array(outcomes)

def streakify_unferr(points1, points2):
    # Subset points to where at least a return was made
    points1 = points1[(points1.P1DoubleFault != 1) & (points1.P2DoubleFault != 1)]
    points1 = points1[points1.P2UnfErr == 0]

    points2 = points2[(points2.P1DoubleFault != 1) & (points2.P2DoubleFault != 1)]
    points2 = points2[points2.P1UnfErr == 0]

    outcomes = []

    for i in range(np.shape(points1)[0]):
        isErr = points1.iloc[i].P1UnfErr
        whosPoint = points1.iloc[i].PointWinner
        if isErr:
            outcomes.append(0)
        elif whosPoint == 1:
            outcomes.append(1)

    for i in range(np.shape(points2)[0]):
        isErr = points2.iloc[i].P2UnfErr
        whosPoint = points2.iloc[i].PointWinner
        if isErr:
            outcomes.append(0)
        elif whosPoint == 2:
            outcomes.append(1)
        
    return np.array(outcomes)

def streakify_serves(points1, points2):
    # Subset to points where player is serving
    points1 = points1[points1.PointServer == 1]
    points2 = points2[points2.PointServer == 2]

    outcomes = []

    for i in range(np.shape(points1)[0]):
        if points1.iloc[i].P1Ace == 1:
            outcomes.append(1)
        else:
            outcomes.append(0)

    for i in range(np.shape(points2)[0]):
        if points2.iloc[i].P2Ace == 1:
            outcomes.append(1)
        else:
            outcomes.append(0)
        
    return np.array(outcomes)

#--------------------------------------------------------------------
# a
#--------------------------------------------------------------------

# Import ATP Mens top 100
matchups100 = pd.read_csv('matchups_atp100.csv')
players = matchups100.columns[1:]

colnames = ['Player', 'kmax','chi2','p-val']
player_stats = pd.DataFrame(0, index = players, columns=colnames)

tours = ['ausopen', 'frenchopen', 'usopen', 'wimbledon']


for player in tqdm(players):
    outcomes = np.array([])
    for year in np.arange(2014, 2018):
        for tour in tours:
            matches = pd.read_csv('tennis_data/' + str(year) + '-' + tour + '-matches.csv')
            points = pd.read_csv('tennis_data/' + str(year) + '-' + tour + '-points.csv')

            tour_players = list(set(np.append(matches.player1.values, matches.player2.values)))

            # Go through tournament players only in the top 100 (avoids NaN values)
            if player in tour_players:
                points1, points2 = get_player_points(player, matches, points)
                outcomes = np.append(outcomes, streakify_serves(points1, points2))
            
            # Player not in this tour so pass
            else:
                pass
    

    make,miss = find_streak_distributions(outcomes,100)
    mu_make,std_make,mu_miss,std_miss = simulated_iid_series(outcomes,kmax=100)
    kmax, chi2, pval = test_streak_distribution_hypothesis(make,mu_make,std_make)


    player_stats.loc[player] = [player, kmax, chi2, pval]


# player_stats = player_stats.reset_index()
player_stats.to_csv('streaks_serves.csv', index=False)