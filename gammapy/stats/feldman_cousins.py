#!/usr/bin/python

import numpy, pylab
from scipy import stats

################################################################################
# Construct confidence belts                                                   #
################################################################################

#def construct_confidence_belt(distribution_dict, bins, alpha):
#
# Function to choose bins a la Feldman Cousins ordering principel.
#
# These are the arguments:
#     distribution_dict = a dictionary of mu values and a corresponding list of x values
#     bins              = the bins the x distribution will have
#     alpha             = your desired confidence level

def construct_confidence_belt(distribution_dict, bins, alpha):

    distributions_scaled = []

    # Histogram gets rid of the last bin, so add one extra
    bin_width = bins[1] - bins[0]
    new_bins = numpy.concatenate((bins, numpy.array([bins[-1]+bin_width])), axis=0)

    # Histogram and normalise each distribution so it is a real PDF
    for mu, distribution in iter(sorted(distribution_dict.iteritems())):
        entries = numpy.histogram(distribution, bins=new_bins)[0]
        integral = float(sum(entries))
        distributions_scaled.append(entries/integral)

    rank, confidence_belt = construct_confidence_belt_PDFs(distributions_scaled, alpha)

    return rank, confidence_belt

################################################################################
# Function that actually constructs the confidence belts                       #
################################################################################

#def construct_confidence_belt_PDFs(distributions_scaled, alpha):
#
# Function to choose bins a la Feldman Cousins ordering principel.
#
# These are the arguments:
#     distribution_scaled = a matrix of mu values and PDFs
#     alpha = your desired confidence level

def construct_confidence_belt_PDFs(matrix, alpha):

    number_mus    = len(matrix)
    number_x_bins = len(matrix[0])

    #mu_to_plot = 151

    #print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % \
                                             #(round(distributions_scaled[mu_to_plot][0],3),
                                             #round(distributions_scaled[mu_to_plot][1],3),
                                             #round(distributions_scaled[mu_to_plot][2],3),
                                             #round(distributions_scaled[mu_to_plot][3],3),
                                             #round(distributions_scaled[mu_to_plot][4],3),
                                             #round(distributions_scaled[mu_to_plot][5],3),
                                             #round(distributions_scaled[mu_to_plot][6],3),
                                             #round(distributions_scaled[mu_to_plot][7],3),
                                             #round(distributions_scaled[mu_to_plot][8],3),
                                             #round(distributions_scaled[mu_to_plot][9],3),
                                             #round(distributions_scaled[mu_to_plot][10],3),
                                             #round(distributions_scaled[mu_to_plot][11],3)))

    distributions_scaled    = numpy.array(matrix)
    distributions_re_scaled = numpy.array(matrix)
    summed_propability      = numpy.zeros(number_mus)

    # Step 1:
    # For each x, find the greatest likelihood in the mu direction.
    # greatest_likelihood is an array of length number_x_bins.
    greatest_likelihood = numpy.amax(distributions_scaled, axis=0)

    #print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % \
                                             #(round(greatest_likelihood[0],3),
                                             #round(greatest_likelihood[1],3),
                                             #round(greatest_likelihood[2],3),
                                             #round(greatest_likelihood[3],3),
                                             #round(greatest_likelihood[4],3),
                                             #round(greatest_likelihood[5],3),
                                             #round(greatest_likelihood[6],3),
                                             #round(greatest_likelihood[7],3),
                                             #round(greatest_likelihood[8],3),
                                             #round(greatest_likelihood[9],3),
                                             #round(greatest_likelihood[10],3),
                                             #round(greatest_likelihood[11],3)))

    #print greatest_likelihood[0]

    # Set to same value if none of the bins has an entry to avoid division by
    # zero
    greatest_likelihood[greatest_likelihood == 0] = 1

    #print greatest_likelihood[0]

    # Step 2:
    # Scale all entries by this value
    distributions_re_scaled /= greatest_likelihood

    #print distributions_re_scaled[0]
    #print distributions_re_scaled[1]

    # Step 3 (Feldman Cousins Ordering principel):
    # For each mu, get the largest entry
    largest_entry = numpy.argmax(distributions_re_scaled, axis = 1)
    # Set the rank to 1 and add probability
    for i in xrange(number_mus):
        distributions_re_scaled[i][largest_entry[i]] = 1
        summed_propability[i]  += numpy.sum(numpy.where(distributions_re_scaled[i] == 1, distributions_scaled[i], 0))
        distributions_scaled[i] = numpy.where(distributions_re_scaled[i] == 1, 1, distributions_scaled[i])

    #print distributions_re_scaled[0]
    #print distributions_re_scaled[1]

    # Identify next largest entry not yet ranked. While there are entries
    # smaller than 1, some bins don't have a rank yet.
    while numpy.amin(distributions_re_scaled) < 1:
        # For each mu, this is the largest rank attributed so far.
        largest_rank = numpy.amax(distributions_re_scaled, axis=1)
        # For each mu, this is the largest entry that is not yet a rank.
        largest_entry = numpy.where(distributions_re_scaled < 1, distributions_re_scaled, -1)
        #print largest_entry[0]
        # For each mu, this is the position of the largest entry that is not yet a rank.
        largest_entry_position = numpy.argmax(largest_entry, axis = 1)
        # Invalidate indices where there is no maximum (every entry is already a rank)
        largest_entry_position = [largest_entry_position[i] if largest_entry[i][largest_entry_position[i]] != -1 else -1 for i in xrange(len(largest_entry_position))]
        # Replace the largest entry with the highest rank so far plus one
        # Add the probability
        for i in xrange(number_mus):
            if largest_entry_position[i] == -1:
                continue
            distributions_re_scaled[i][largest_entry_position[i]] = largest_rank[i] + 1
            if summed_propability[i] < alpha:
                summed_propability[i] += distributions_scaled[i][largest_entry_position[i]]
                distributions_scaled[i][largest_entry_position[i]] = 1
            else:
                distributions_scaled[i][largest_entry_position[i]] = 0

    #print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % \
                                             #(round(distributions_re_scaled[mu_to_plot][0],3),
                                             #round(distributions_re_scaled[mu_to_plot][1],3),
                                             #round(distributions_re_scaled[mu_to_plot][2],3),
                                             #round(distributions_re_scaled[mu_to_plot][3],3),
                                             #round(distributions_re_scaled[mu_to_plot][4],3),
                                             #round(distributions_re_scaled[mu_to_plot][5],3),
                                             #round(distributions_re_scaled[mu_to_plot][6],3),
                                             #round(distributions_re_scaled[mu_to_plot][7],3),
                                             #round(distributions_re_scaled[mu_to_plot][8],3),
                                             #round(distributions_re_scaled[mu_to_plot][9],3),
                                             #round(distributions_re_scaled[mu_to_plot][10],3),
                                             #round(distributions_re_scaled[mu_to_plot][11],3)))

    #print distributions_re_scaled[0]
    #print distributions_re_scaled[1]
    #print distributions_scaled[0]
    #print distributions_scaled[1]

    ##This is the slow, not optimised code version.

    #for x in xrange(number_x_bins):
       ## Seaching greatest_likelihood
       #greatest_likelihood = 0
       #for i in xrange(number_mus):
           #if greatest_likelihood < distributions_scaled[i][x]:
               #greatest_likelihood = distributions_scaled[i][x]
       ## Re-scaling
       #for i in xrange(number_mus):
           #if greatest_likelihood != 0:
               #distributions_re_scaled[i][x] = distributions_scaled[i][x] / greatest_likelihood
           #else:
               #distributions_re_scaled[i][x] = 0

    #for i in xrange(number_mus):
       #for position_of_next_smaller_entry in xrange(1, number_x_bins + 1):
           #maximum = 0
           #for x in xrange(number_x_bins):
               #if distributions_re_scaled[i][x] > maximum and distributions_re_scaled[i][x] < 1:
                   #maximum = distributions_re_scaled[i][x]
           #for x in xrange(number_x_bins):
               #if distributions_re_scaled[i][x] == maximum:
                   #distributions_re_scaled[i][x] = position_of_next_smaller_entry

    #for i in xrange(number_mus):
        #mu_distribution = distributions_re_scaled[i]
        #scaled_distribution = distributions_scaled[i]
        #prob = 0
        #number = 0
        #while prob < alpha:
            #number += 1
            #if number > number_x_bins:
                #print "Did not reach requiered significant level!"
                #break
            ##indices_to_add = numpy.where(mu_distribution == number, scaled_distribution, 0)
            ##prob += numpy.sum(indices_to_add)
            #for x in xrange(number_x_bins):
                #if number == distributions_re_scaled[i][x]:
                    #prob += distributions_scaled[i][x]
        ##if prob - alpha > 0.02:
            ##print prob
            ##print mus[i]
            ##print "Warning! Coverage more than 2 per cent wrong!"
        ##distributions_re_scaled[i] = numpy.where(distributions_re_scaled[i] < number, 0, distributions_re_scaled[i])
        #for x in xrange(number_x_bins):
            #if distributions_re_scaled[i][x] <= number:
                #distributions_re_scaled[i][x] = 0

    return distributions_re_scaled, distributions_scaled

################################################################################
# Find upper and lower limit from confidence interval                          #
################################################################################

def GetUpperAndLowerLimit(mu_bins, x_bins, distributions_scaled, do_plot = False):

    upper_limit = []
    lower_limit = []

    number_mu     = len(mu_bins)
    number_bins_x = len(x_bins)

    for mu in xrange(number_mu):
        x_values = []
        upper_limit.append(-1)
        lower_limit.append(-1)
        for x in xrange(number_bins_x):
            #This point lies in the confidence interval
            if distributions_scaled[mu][x] == 1:
                x_value = x_bins[x]
                x_values.append(x_value)
                # Upper limit is the first point where this condition is true
                if upper_limit[-1] == -1:
                    upper_limit[-1] = x_value
                # Lower limit is the first point after this condition is not true
                if x == number_bins_x - 1:
                    lower_limit[-1] = x_value
                else:
                    lower_limit[-1] = x_bins[x + 1]
        if do_plot:
            pylab.plot(x_values, [mu_bins[mu] for i in xrange(len(x_values))], marker='.', ls='',color='black')

    return upper_limit, lower_limit

################################################################################
# Push limits outwards as described in the FC paper                            #
################################################################################

def FixUpperAndLowerLimit(upper_limit, lower_limit):

    all_fixed = False

    while not all_fixed:
        all_fixed = True
        for j in xrange(1,len(upper_limit)):
            if upper_limit[j] < upper_limit[j-1]:
                #print "I had to move out an upper limit!"
                #print j
                #print str(upper_limit[j]) + " " + str(upper_limit[j-1])
                upper_limit[j-1] = upper_limit[j]
                all_fixed = False
        for j in xrange(1,len(lower_limit)):
            if lower_limit[j] < lower_limit[j-1]:
                #print "I had to move out a lower limit!"
                #print str(lower_limit[j]) + " " + str(lower_limit[j-1])
                lower_limit[j] = lower_limit[j-1]
                all_fixed = False

################################################################################
# Find the upper limit for a given x value                                     #
################################################################################

def FindLimit(x_value, x_values_input, y_values_input, do_upper_limit = True):

    limit = 0

    if do_upper_limit:
        previous_x = numpy.nan
        next_value = False
        identical = True
        x_values = x_values_input
        y_values = y_values_input
        for i in xrange(len(x_values)):
            current_x = x_values[i]
            # If the x_value did lie on the bin border, loop until the x value
            # is changing and take the last point (that is the highest point in
            # case points lie on top of each other.
            if next_value == True and current_x != previous_x:
                limit = y_values[i-1]
                break
            if x_value <= current_x:
                # If the x_value does not lie on the bin border, this should be
                # the upper limit
                if x_value != current_x:
                    limit = y_values[i]
                    break
                next_value = True
            previous_x = current_x
    else:
        x_values = numpy.flipud(x_values_input)
        y_values = numpy.flipud(y_values_input)
        for i in xrange(len(x_values)):
            current_x = x_values[i]
            if x_value >= current_x:
                limit = y_values[i]
                break

    return limit

################################################################################
# Calculate the average upper limit                                            #
################################################################################

def FindAverageUpperLimit(x_bins, distributions_scaled, upper_limit, mu_bins):

    avergage_limit = 0
    number_points = len(distributions_scaled[0])*1.0

    for i in xrange(number_points):
        avergage_limit += distributions_scaled[0][i]*FindLimit(x_bins[i], upper_limit, mu_bins)

    return avergage_limit

################################################################################
# Find Confidence Interval assuming a Gaussian                                 #
################################################################################

def FindConfidenceIntervalGauss(mu, sigma, x_bins, fCL):

    dist = stats.norm(loc=mu, scale=sigma)

    x_bin_width = x_bins[1] - x_bins[0]

    p = []
    r = []

    for x in x_bins:
        #print x
        p.append(dist.pdf(x)*x_bin_width)
        # This is the formula from the FC paper
        if mu == 0 and sigma == 1:
            if x < 0:
                r.append(numpy.exp(x*mu-mu*mu*0.5))
            else:
                r.append(numpy.exp(-0.5*numpy.power((x-mu),2)))
        # This is the more general formula
        else:
            # Implementing the boundary condition at zero
            muBest     = max(0, x)
            probMuBest = stats.norm.pdf(x, loc=muBest, scale=sigma)
            if probMuBest == 0.0:
              r.append(0.0);
            else:
              r.append(p[-1]/probMuBest)
        #print r[-1]

    p = numpy.asarray(p)
    r = numpy.asarray(r)

    if sum(p) < fCL:
        print "Bad choice of x-range for this mu!"
        print "Not enough probability in x bins to reach confidence level!"

    rank = stats.rankdata(-r, method='dense')

    #print p
    #print r
    #print rank

    index_array = numpy.arange(x_bins.size)

    rank_sorted, index_array_sorted = zip(*sorted(zip(rank, index_array)))

    #print index_array_sorted
    #print rank_sorted

    index_min = index_array_sorted[0]
    index_max = index_array_sorted[0]

    p_sum = 0

    for i in xrange(len(rank_sorted)):
        #print index_min
        #print index_max
        #print index_array_sorted[i]
        if index_array_sorted[i] < index_min:
            index_min = index_array_sorted[i]
        if index_array_sorted[i] > index_max:
            index_max = index_array_sorted[i]
        #if i < len(rank_sorted) - 1 and rank_sorted[i] == rank_sorted[i+1]:
            #print rank_sorted[i]
            #print rank_sorted[i+1]
            #continue
        #print index_min
        #print index_max
        p_sum += p[index_array_sorted[i]]
        #p_sum = sum(p[index_min:index_max+1])
        #print p_sum
        if p_sum >= fCL:
            break

    #print p_sum

    #print index_min
    #print index_max

    return [x_bins[index_min], x_bins[index_max] + x_bin_width]

################################################################################
# Find Confidence Interval assuming a Poisson distribution                     #
################################################################################

def FindConfidenceIntervalPoisson(mu, background, x_bins, fCL):

    dist = stats.poisson(mu=mu+background)

    x_bin_width = x_bins[1] - x_bins[0]

    p = []
    r = []

    for x in x_bins:
        #print x
        p.append(dist.pmf(x))
        # Implementing the boundary condition at zero
        muBest = max(0, x - background)
        probMuBest = stats.poisson.pmf(x, mu=muBest+background)
        if probMuBest == 0.0:
          r.append(0.0);
        else:
          r.append(p[-1]/probMuBest)
        #print r[-1]

    p = numpy.asarray(p)
    r = numpy.asarray(r)

    if sum(p) < fCL:
        print "Bad choice of x-range for this mu!"
        print "Not enough probability in x bins to reach confidence level!"

    rank = stats.rankdata(-r, method='dense')

    #print p
    #print r
    #print rank

    index_array = numpy.arange(x_bins.size)

    rank_sorted, index_array_sorted = zip(*sorted(zip(rank, index_array)))

    #print index_array_sorted
    #print rank_sorted

    index_min = index_array_sorted[0]
    index_max = index_array_sorted[0]

    p_sum = 0

    for i in xrange(len(rank_sorted)):
        #print index_min
        #print index_max
        #print index_array_sorted[i]
        if index_array_sorted[i] < index_min:
            index_min = index_array_sorted[i]
        if index_array_sorted[i] > index_max:
            index_max = index_array_sorted[i]
        #if i < len(rank_sorted) - 1 and rank_sorted[i] == rank_sorted[i+1]:
            #print rank_sorted[i]
            #print rank_sorted[i+1]
        #    continue
        #print index_min
        #print index_max
        p_sum += p[index_array_sorted[i]]
        #p_sum = sum(p[index_min:index_max+1])
        #print p_sum
        if p_sum >= fCL:
            break

    #print p_sum

    #print index_min
    #print index_max

    return x_bins[index_min], x_bins[index_max] + x_bin_width

################################################################################
# Make it usable as script                                                     #
################################################################################

if __name__ == "__main__":
    print "No tests implemented :("