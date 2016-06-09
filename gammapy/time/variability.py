def exptest_for_run2(time_delta=[]):
    """

    Short description: To compute the Mr value, which shows the level of variability for a certain period of time
    Longer description: A single Mr value can be calculated, which shows the level of variability for the whole period, or the Mr value for each run can be shown.


    Ref:Prah(1999).A fast unbinned test on event clustering in Poisson processes.astro-ph/9909399

    Parameters
    ----------
    run: run number for each event
    event_time : list of times for each event
    expCount: the acceptance for each run according to the observation conditions

     Returns
    -------
    Mr : float
        Level of variability
    """
    mean_time = np.mean(time_delta)
    normalized_time_delta = time_delta / mean_time
    sum_time = []
    for i in range(len(normalized_time_delta)):
        if normalized_time_delta[i] < 1:
            sum_time.append(1 - normalized_time_delta[i] / 1.0)
    mean_normalized_time=np.mean(normalized_time_delta)
    M_value=0
    Mr=0
    sum_time_all = np.sum([sum_time])
    if len(time_delta)!=0:
        M_value = sum_time_all / len(time_delta)
        Mr = (M_value - (1 / 2.71828 - 0.189 / len(time_delta))) / (0.2427 / math.sqrt(len(time_delta)))
    return Mr
