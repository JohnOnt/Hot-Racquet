import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from tqdm import tqdm

#--------------------------------------------------------------------
# Data Helper Functions
#--------------------------------------------------------------------

def get_player_points2(player, matches, points, rankings):
    # Get all match numbers
    match_ids1 = matches[(matches['player1'] == player)].match_id
    match_ids2 = matches[(matches['player2'] == player)].match_id

    player_points1 = []
    player_points2 = []

    for m_id in match_ids1:
        # Check if player they are playing against has an elo
        p2 = matches.player2.values[matches.match_id == m_id][0]
        if (p2 in (rankings.name.values)) == False:
            # Dont add to list
            continue
        player_points1.append(points[points.match_id == m_id])

    for m_id in match_ids2:
        # Check if player they are playing against has an elo
        p1 = matches.player1.values[matches.match_id == m_id][0]
        if (p1 in (rankings.name.values)) == False:
            # Dont add to list
            continue
        player_points2.append(points[points.match_id == m_id])

    return player_points1, player_points2

def get_metadata(match, matches, rankings, px):
    m_id = match.iloc[0].match_id
    p1 = matches.player1.values[matches.match_id == m_id][0]
    p2 = matches.player2.values[matches.match_id == m_id][0]

    elo1 = rankings.points[rankings.name == p1].values[0]
    elo2 = rankings.points[rankings.name == p2].values[0]

    # Get number of times player serves and recieves for each position
    if px == 1:
        nserves = np.shape(match[match.PointServer == 1])[0]
        nrecs   = np.shape(match[match.PointServer == 2])[0]

        return elo1, elo2, nserves, nrecs
    
    else:
        nserves = np.shape(match[match.PointServer == 2])[0]
        nrecs   = np.shape(match[match.PointServer == 1])[0]

        return elo1, elo2, nserves, nrecs

    
#--------------------------------------------------------------------
# Load Point Probability Model
#--------------------------------------------------------------------

lm_points = pickle.load(open('point_prob_model.sav', 'rb'))

#--------------------------------------------------------------------
# Streak Distribution Functions
#--------------------------------------------------------------------

def simulated_iid_series2(metadata1, metadata2, nsims=1000,kmax = 15):
    # calculate the number of streaks of a given length for an IDD series with P_M = mean of input series
    tmake = []
    tmiss = []
    for meta_i in metadata1:
        # meta_i = metadata1[i]
        nserves = meta_i[2]
        nrecs = meta_i[3]

        pserve = lm_points.predict_proba(np.array([[(meta_i[0] - meta_i[1]), meta_i[4]]]))[:,0][0]
        prec = lm_points.predict_proba(np.array([[(meta_i[0] - meta_i[1]), meta_i[4]]]))[:,1][0]

        # For all receiving points
        for _ in range(nsims):
            x_serves = stats.bernoulli.rvs(pserve, size=nserves)
            x_recs = stats.bernoulli.rvs(prec, size=nrecs)
            make_serves, miss_serves = find_streak_distributions(x_serves,kmax)
            make_recs, miss_recs = find_streak_distributions(x_recs,kmax)

            tmake.append(make_serves)
            tmake.append(make_recs)
            tmiss.append(miss_serves)
            tmiss.append(miss_recs)

    for meta_i in metadata2:
        nserves = meta_i[2]
        nrecs = meta_i[3]

        pserve = lm_points.predict_proba(np.array([[(meta_i[1] - meta_i[0]), meta_i[4]]]))[:,0][0]
        prec = lm_points.predict_proba(np.array([[(meta_i[1] - meta_i[0]), meta_i[4]]]))[:,1][0]

        # For all receiving points
        for _ in range(nsims):
            x_serves = stats.bernoulli.rvs(pserve, size=nserves)
            x_recs = stats.bernoulli.rvs(prec, size=nrecs)
            make_serves, miss_serves = find_streak_distributions(x_serves,kmax)
            make_recs, miss_recs = find_streak_distributions(x_recs,kmax)

            tmake.append(make_serves)
            tmake.append(make_recs)
            tmiss.append(miss_serves)
            tmiss.append(miss_recs)

    tmiss = np.array(tmiss)
    tmake = np.array(tmake)
    # Averaged not counting the different matches
    mu_make = np.sum(tmake, axis=0) / nsims
    mu_miss = np.sum(tmiss, axis=0) / nsims
    # also adjusted standard deviation
    Nmatches = (np.shape(metadata1)[0] + np.shape(metadata2)[0])
    sig_make = np.std(tmake,axis=0) * np.sqrt(Nmatches)
    sig_miss = np.std(tmiss,axis=0) * np.sqrt(Nmatches)

    return mu_make,sig_make,mu_miss,sig_miss

def test_streak_distribution_hypothesis(counts,mu,sig,null_only=False):
    # takes input from find_streak_distributions and simulated_series
    if np.sum(counts) == 0:
        # Player has no counts of streaks so must return NA values
        return np.nan, np.nan, np.nan

    kmax = 15 # np.max(np.nonzero(counts))+1
    chi2 = np.sum((counts[:kmax]-mu[:kmax])**2/sig[:kmax]**2)
    pval = 1-stats.chi2.cdf(chi2,kmax-1)

    return kmax, chi2, pval

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

def find_streak_distributions2(outcomes, run_length):
    # find number of streaks of misses or makes up to a length 'run_length'
    make_counts = np.zeros(run_length)
    miss_counts = np.zeros(run_length)
    for x in outcomes:
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

def streakify_points2(points, px):
    outcomes = []

    for i in range(np.shape(points)[0]):
        whosPoint = points.iloc[i].PointWinner
        if whosPoint == px:
            outcomes.append(1)
        else:
            outcomes.append(0)

    return np.array(outcomes)


#--------------------------------------------------------------------
# Main Part
#--------------------------------------------------------------------

# Import ATP Mens top 100
rankings = pd.read_csv('Elo_Rankings2017.csv')
# players = ['Roger Federer']
players = rankings.name[:101]

colnames = ['Player', 'kmax','chi2','p-val']
player_stats = pd.DataFrame(0, index = players, columns=colnames)

tours = ['ausopen', 'frenchopen', 'usopen', 'wimbledon']
# tours = ['ausopen', 'usopen']

for player in tqdm(players):
    outcomes = []
    metadata1 = []
    metadata2 = []
    for year in np.arange(2014, 2018):
        for tour in tours:
            matches = pd.read_csv('tennis_data/' + str(year) + '-' + tour + '-matches.csv')
            points = pd.read_csv('tennis_data/' + str(year) + '-' + tour + '-points.csv')

            # Establish the surface
            if tour == 'wimbledon':
                court = 1 #'grass'
            elif tour == 'frenchopen':
                court = 2 # 'clay'
            else:
                court = 3 # 'hard'

            tour_players = list(set(np.append(matches.player1.values, matches.player2.values)))

            # Go through tournament players only in the top 100 (avoids NaN values)
            if player in tour_players:

                # these are lists now
                points1, points2 = get_player_points2(player, matches, points, rankings)

                for match in points1:
                    outcomes.append(streakify_points2(match, 1))
                    # Need elo of both players and court
                    elo1, elo2, nserves, nrecs = get_metadata(match, matches, rankings, 1)
                    metadata1.append([elo1, elo2, nserves, nrecs, court])

                for match in points2:
                    outcomes.append(streakify_points2(match, 2))
                    # Need elo of both players and court
                    elo1, elo2, nserves, nrecs = get_metadata(match, matches, rankings, 2)
                    metadata2.append([elo1, elo2, nserves, nrecs, court])
            
            # Player not in this tour so pass
            else:
                pass
    

    
    make,miss = find_streak_distributions2(outcomes,15)
    mu_make,std_make,mu_miss,std_miss = simulated_iid_series2(metadata1, metadata2, kmax=15)
    kmax, chi2, pval = test_streak_distribution_hypothesis(make,mu_make,std_make)

    # Save the data for the GOAT
    # if player == 'Roger Federer':
        # mu = mu_make
        # sig = std_make
        # counts = make
        # federer_data = {'mu': mu_make,
                        # 'sig': std_make,
                        # 'counts': make,
                        # 'kmax': kmax}
        
        # pickle.dump(federer_data, open('federer_data.sav', 'wb'))


    player_stats.loc[player] = [player, kmax, chi2, pval]


print(player_stats)
player_stats = player_stats.reset_index()
player_stats.to_csv('streaks_points.csv', index=False)