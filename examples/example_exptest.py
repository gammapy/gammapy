def exptest(table):
    """
    An example showing how to use gammapy/time/exp_test_for_run

    Parameters
    ----------
    table : astropy.table as the input

    plot : This function returns a histogram for the distribution of the Mr value of each run.
           A normal distribution is expected for non-flaring sources.  
    """
    size=len(table['runnum'])
    individual_run=[]
    num_of_events = []
    M_value=[]
    time_delta_all=[]
  
   #number of runs in the list
    for i in range(0,size-1):
        if(table['runnum'][i]!=table['runnum'][i+1]):
            individual_run.append(table['runnum'][i])
    individual_run.append(table['runnum'][size-1])
    num_of_runs=len(individual_run)
 
   #number of events in a run
    for i in range(0, len(individual_run)):
        num_of_events_temp=0
        for j in range(0,size):
            if (individual_run[i] == table['runnum'][j]):
                num_of_events_temp=num_of_events_temp+1
        num_of_events.append(num_of_events_temp)

   #start here,calculation of the Mr value for each run, and for the whole dataset
    for i in range(0,num_of_runs):
        time_delta_each_run = []
        for j in range(0,size-1):
            if (num_of_events[i]>20 and individual_run[i]==table['runnum'][j] and table['runnum'][j]==table['runnum'][j+1]):
                time_delta_each_run.append((table['mjd'][j + 1] - table['mjd'][j]) * 0.5 * (table['expCount'][j + 1] + table['expCount'][j]))
                time_delta_all.append((table['mjd'][j + 1] - table['mjd'][j]) * 0.5 * (table['expCount'][j + 1] + table['expCount'][j]))
        M = exptest_for_run(time_delta_each_run)
        M_value.append(M)
        del time_delta_each_run[:]
        overallM = exptest_for_run(time_delta_all)
  
    return M_value
    

def plot(M_value=[]):
    (mu, sigma) = norm.fit(M_value)
    n, bins, patches = plt.hist(M_value, bins=30, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    print(mu, sigma)
    plt.xlabel('Mr value')
    plt.ylabel('counts')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.grid(True)
    plt.show()

def readinfile():

    table = Table.read('list.txt', format='ascii')
    return table

if __name__ == '__main__':
     table=readinfile()
     M_value=exptest(table)
     plot(M_value)
