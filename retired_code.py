
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