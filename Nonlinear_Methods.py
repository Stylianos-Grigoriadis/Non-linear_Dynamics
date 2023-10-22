import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy.matlib as matlib
import sys
from skimage.transform import rotate
from sklearn.metrics import pairwise_distances
from scipy import stats as st
from scipy.spatial import distance as spd
import warnings, sys, string

def Time_delay(data, limit_of_time_lag, Signal_name, n_bins=0, ):
    # Start of Stergiou code
    """
        inputs    - data, column oriented time series
                  - L, maximal lag to which AMI will be calculated
                  - bins, number of bins to use in the calculation, if empty an
                    adaptive formula will be used
                  - to_matlab, an option for MATLAB users of the code, if MATLAB
                    datatypes are needed for output, use this to have them
                    returned with proper types. Default is false.

                    Only use if you have 'matlab.engine' installed in your current
                    Python env.

                    Note: this cannot be installed through the usual conda or pip
                    commands, search online to view resources to help in installing
                    'matlab.engine' for Python.

        outputs   - tau, first minimum in the AMI vs lag plot
                  - v_AMI, vector of AMI values and associated lags

        inputs    - x, single column array with the same length as y.
                  - y, single column array with the same length as x.
        outputs   - ami, the average mutual information between the two arrays

        Remarks
        - This code uses average mutual information to find an appropriate lag
          with which to perform phase space reconstruction. It is based on a
          histogram method of calculating AMI.
        - In the case a value of atu could not be found before L the code will
          automatically re-execute with a higher value of L, and will continue to
          re-execute up to a ceiling value of L.

        Future Work
        - None currently.

        Mar 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
                  - Modified code to output a plot and notify the user if a value
                    of tau could not be found.
        Sep 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
                  - Previously the number of bins was hard coded at 128. This
                    created a large amount of error in calculated AMI value and
                    vastly decreased the sensitivity of the calculation to changes
                    in lag. The number of bins was replaced with an adaptive
                    formula well known in statistics. (Scott 1979
                  - The previous plot output was removed.
        Oct 2017 - Modified by Ben Senderling, email unonbcf@unomaha.edu
                  - Added print commands to display progress.
        May 2019 - Modified by Ben Senderling, email unonbcf@unomaha.edu
                  - In cases where L was not high enough to find a minimun the
                    code would reexecute with a higher L, and the binned data.
                    This second part is incorrect and was corrected by using
                    data2.
                  - The reexecution part did not have the correct input
                    parameters.
        Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
        Variability, University of Nebraska at Omaha

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

        1. Redistributions of source code must retain the above copyright notice,
            this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright
            notice, this list of conditions and the following disclaimer in the
            documentation and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its
            contributors may be used to endorse or promote products derived from
            this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
        IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
        THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
        PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
        CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
        LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
        NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """
    eps = np.finfo(float).eps  # smallest floating point value

    if isinstance(limit_of_time_lag, int):
        N = len(data)

        data = np.array(data)

        if n_bins == 0:
            bins = np.ceil((np.max(data) - np.min(data)) / (3.49 * np.nanstd(data * N ** (-1 / 3), axis=0)))
        else:
            bins = n_bins

        bins = int(bins)

        data = data - min(data)  # make all data points positive
        y = np.floor(data / (np.max(data) / (bins - eps)))
        y = np.array(y,
                     dtype=int)  # converts the vector of double vals from data2 into a list of integers from 0 to overlap (where overlap is N-L).

        v = np.zeros((limit_of_time_lag, 1))  # preallocate the vector
        overlap = N - limit_of_time_lag
        increment = 1 / overlap

        pA = sp.csr_matrix((np.full(overlap, increment), (y[0:overlap], np.ones(overlap, dtype=int)))).toarray()[:, 1]

        v = np.zeros((2, limit_of_time_lag))

        for lag in range(limit_of_time_lag):  # used to be from 0:L-1 (BS)
            v[0, lag] = lag

            pB = sp.csr_matrix(
                (np.full(overlap, increment), (y[lag:overlap + lag], np.ones(overlap, dtype=int)))).toarray()[:, 1]
            # find joint probability p(A,B)=p(x(t),x(t+time_lag))
            pAB = sp.csr_matrix((np.full(overlap, increment), (y[0:overlap], y[lag:overlap + lag])))

            (A, B) = np.nonzero(pAB)
            AB = pAB.data

            v[1, lag] = np.sum(
                np.multiply(AB, np.log2(np.divide(AB, np.multiply(pA[A], pB[B])))))  # Average Mutual Information

        tau = np.array(np.full((limit_of_time_lag, 2), -1, dtype=float))

        j = 0
        for i in range(v.shape[1] - 1):  # Finds first minimum
            if v[1, i - 1] >= v[1, i] and v[1, i] <= v[1, i + 1]:
                ami = v[1, i]
                tau[j, :] = np.array([i, ami])
                j += 1

        tau = tau[:j]  # only include filled in data.

        initial_AMI = v[1, 0]
        for i in range(v.shape[1]):  # Finds first AMI value that is 20% initial AMI
            if v[1, i] < (0.2 * initial_AMI):
                tau[0, 1] = i
                break

        v_AMI = v
        ami = v_AMI


    elif isinstance(limit_of_time_lag, np.ndarray) or isinstance(limit_of_time_lag, list):
        x = data if isinstance(data, np.ndarray) else np.array(data)
        y = limit_of_time_lag if isinstance(limit_of_time_lag, np.ndarray) else np.array(limit_of_time_lag)

        if len(x) != len(y):
            raise ValueError('X and Y must be the same size.')

        increment = 1 / len(y)
        one = np.ones(len(y), dtype=int)

        bins1 = np.ceil((max(x) - min(x)) / (3.49 * np.nanstd(x) * len(x) ** (-1 / 3)))  # Scott 1979
        bins2 = np.ceil((max(y) - min(y)) / (3.49 * np.nanstd(y) * len(y) ** (-1 / 3)))  # Scott 1979
        x = x - min(x)  # make all data points positive
        x = np.floor(x / (max(x) / (bins1 - eps)))  # scaling the data
        y = y - min(y)  # make all data points positive
        y = np.floor(y / (max(y) / (bins2 - eps)))  # scaling the data

        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)

        increment = np.full(len(y), increment)
        pA = sp.csr_matrix((increment, (x, one))).toarray()[:, 1]
        pB = sp.csr_matrix((increment, (y, one))).toarray()[:, 1]
        pAB = sp.csr_matrix((increment, (x, y)))
        (A, B) = np.nonzero(pAB)
        AB = pAB.data
        ami = np.sum(np.multiply(AB, np.log2(np.divide(AB, np.multiply(pA[A], pB[B])))))

    else:
        raise ValueError('Invalid input, read documentation for input options.')
    # End of Stergiou code
    Time_lag = list(ami[0])
    AMI = list(ami[1])
    #a = list(ami)
    #print(ami)
    #print(ami[0][0])
    #print(ami[0][1])
    #print(ami[0])
    #print(ami[1])
    #Time_lag = list(ami[0])
    #AMI = list(ami[1])
    #print(type(Time_lag))
    #print(type((ami)))
    #print(type((ami[0][0])))
    #print(type((ami[0][1])))
    #print(type((ami[0])))
    #print(type((ami[1])))

    # Time_lag = list(a[1][0])
    # print(type(Time_lag))
    # print(Time_lag)
    # = list(a[1][1])
    min_value = AMI[0]
    min_index = 0
    for i in range(1, len((Time_lag))):
        if AMI[i] < min_value:
            min_value = AMI[i]
            min_index = i
    print("Minimum value of Average mutual information is " + str(min_value))
    print(
        "Time lag at which the Minimum value of Average mutual information appears is " + str(int(Time_lag[min_index])))
    # plt.scatter(Time_lag, AMI)
    # plt.axvline(x=Time_lag[min_index], color='red', label='Time lag at min AMI')
    # plt.ylabel("Average mutual information", fontsize=15)
    # plt.xlabel("Time lag", fontsize=15)
    # plt.legend()
    # plt.title(Signal_name, fontsize=20)
    # plt.show()
    return int(Time_lag[min_index])


def Culculation_of_embending_dimensions(data, tau, MaxDim, speed, Signal_name,  Rtol = 15, Atol = 2):
    """
          data - column oriented time series
          tau - time delay
          MaxDim - maximum embedding dimension
          Rtol - threshold for the first criterion
          Atol - threshold for teh second criterion
          speed - a 0 for the code to calculate to the MaxDim or a 1 for the code
                  to finish once a minimum is found
        Remarks
        - This code determines the embedding dimension for a time series using
          the false nearest neighbors method.
        - Recommended values are Rtol=15 and Atol=2;
        - Reference:   "Determining embedding dimension for phase-space
                        reconstruction using a geometrical construction",
                        M. B. Kennel, R. Brown, and H.D.I. Abarbanel,
                        Physical Review A, Vol 45, No 6, 15 March 1992,
                        pp 3403-3411.
        Future Work
        - Currently there are two methods of detecting a minimal percentage of
          false nearest neighbors. One method checks for a minima or zero
          percentage, the other looks for a limit. Currently only dim is
          returned. This code can be modified to use a comprimise of the two.
        Prior - Created by someone
        Feb 2015 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
                   No changes were made to the algorithm. Checks were added to
                   provide information to the user in case of an error. The two
                   methods described in future work were also modified to work
                   cooperatively. In a previous version the second method (dim)
                   overwrote the first method (dim2).
        Sep 2015 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
                   Previously, dim was found after the for loop, this version has
                   been modified to allow the code to find the minimum as it
                   calculates FNN. This is set within the inputs.
                   The check that was previously put in has been commented out.
        Oct 2015 - Modified by John McCamley, email: unonbcf@unomaha.edu
                 - Embedded other required functions as subroutines.
        Mar 2017 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
                 - Removed global variables in favor of passing the variables
                   from function to function directly. This significantly
                   improved performance. Checked that the calculated percentages
                   of nearest neighbors are the same as the previous version.
        May 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
                 - Added if statement checkeding data orientation.
        Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
                 - Changed indexing throughout so the input data array doesn't
                   need to be reoriented. Changing this sped the code up an
                   average 11% on 10 test signals.
                 - Removed a couple small for loops and replaced with indexed
                   operations. Was also able to remove within function and
                   replaced with a single line of code.
                 - Removed perviously commented out lines of code that were no
                   longer used.
        """
    n = len(data) - tau * MaxDim
    data_array = np.array(data)
    RA = np.std(data_array)

    z = np.array([data_array[0:n]])
    y = np.array([[]])

    m_search = 2

    indx = np.array(np.arange(0, n))
    dim = np.array([])

    dE = np.zeros((MaxDim, 1))

    for j in range(MaxDim):
        # y = np.array([y,z]) # adds additional dimension
        if j == 0:
            y = np.array([np.append(y, z)])
        else:
            y = np.array(np.vstack([y, z]))
        z = np.array([data_array[tau * (j + 1):n + tau * (j + 1)]])
        L = np.zeros((n))

        (y_model, z_model, sort_list, node_list) = kd_part(y, z, 512)

        for i in range(len(indx)):
            yq = np.array(y[:, indx[i]])  # set up next point to look at

            b_upper = np.inf * np.ones(np.size(yq))
            b_lower = np.negative(b_upper)

            pqd = np.inf * np.ones((1, m_search))
            pqr = np.array([])
            pqz = np.array([])
            L_done = 0

            # A couple returning variables are not necessary. Check documentation of kd_search to see what they are.
            (pqd, y_model, z_model, _, _, pqz, _, _, sort_list, node_list) = kd_search(0, m_search, yq, pqd, y_model,
                                                                                       z_model, L_done, pqr, pqz,
                                                                                       b_upper, b_lower, sort_list,
                                                                                       node_list)

            distance = pqz[0] - pqz[1]

            if np.abs(distance) > pqd[1] * Rtol:
                L[i] = 1

            if np.sqrt(pqd[1] ** 2 + distance ** 2) / RA > Atol:
                L[i] = 1

        dE[j] = np.sum(L) / n

        if speed == 1:
            if j >= 2 and ((dE[j - 2] > dE[j - 1] and (dE[j - 1] < dE[j]))):
                dim = j - 1
                break
            if j >= 1 and np.abs(dE[j] - dE[j - 1]) <= 0.001:
                dim = j - 1
                break
            if dE[j] == 0:
                dim = j
                break

    if speed == 0:
        for i in range(len(dE)):
            if np.abs(dE[i - 1] - dE[i]) <= 0.001:
                dim = i - 1
                break
            try:  # In the case of where the code reaches to len(dE) - 1 and attempts to get dE[len(dE)], which causes an IndexError.
                if (dE[i] == 0) or ((dE[i - 1] > dE[i] and (dE[i] < dE[i + 1]))):
                    dim = i
                    break
            except:
                break

    if np.size(dim) == 0:
        dim = MaxDim - 1
        print('No dimension found, dim set to MaxDim\n')

    # +1 because we started from zero and not one. Might change this in the future
    dE = list(dE)
    for i in range(len(dE)):
        dE[i] = dE[i] * 100
    # plt.scatter(range(0, len(dE)), dE, color="red")
    # plt.plot(range(0, len(dE)), dE)
    # plt.ylabel("% of False Nearest Neighbors ", fontsize=15)
    # plt.xlabel("Dimension", fontsize=15)
    # plt.title(Signal_name, fontsize=20)
    # plt.show()
    return (dE, dim + 1)


def kd_part(y_in, z_in, bin_size):
    """
    Create a kd-tree and partitioned database for
    efficiently finding the nearest neighbors to a point
    in a d-dimensional space.

    y_in: original phase space data
    z_in: original phase space data corresponding to y_in
    bin_size: maximum number of distinct points for each bin
    The outputs are placed into global variables used by
    kdsearch and its subroutines.

    The outputs are...
    sort_list(:,1): discriminator: dimension to use in dividing data
    sort_list(:,2): partition: boundary for dividing data
    node_list(i,:): contains data for the i-th partition
    node_list(:,1): 1st element in y of this partition
    node_list(:,2): last element in y of this partition
    node_list(:,3): location in node_list of left branch
    node_list(:,4): location in node_list of right branch
    y_model: phase space data partitioned into a binary tree
    z_model: phase space data corresponding to each y_model point

    Algorithms from:

    "Data Structures for Range Searching", J.L. Bently, J.H. Friedman,
    ACM Computing Surveys, Vol 11, No 4, p 397-409, December 1979

    "An Algorithm for Finding Best Matches in Logarithmic Expected Time",
    J.H. Friedman, J.L. Bentley, R.A. Finkel, ACM Transactions on
    Mathematical Software, Vol 3, No 3, p 209-226, September 1977.

    Mar 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
                  - Formatted.
    """
    y_model = y_in
    z_model = z_in

    # d: dimension of phase space
    # n_y: number of points to put into partitioned database

    (d, n_y) = y_model.shape

    # Set up first node...
    node_list = np.array([[0, n_y, 0, 0]])
    sort_list = np.array([[0, 0]], dtype=float)

    node = 0
    last = 0

    while node <= last:  # check if the node can be divided
        segment = np.array([np.arange(node_list[node, 0], node_list[node, 1])])
        # previously: [node_list[node,1]:node_list[node,2]]
        rg = np.amax(y_model, axis=1) - np.amin(y_model, axis=1)  # previously i and segment were swapped

        # segment.shape[1] is the length of the segment (specifically the length of the row)
        if np.max(rg) > 0 and segment.shape[1] >= bin_size:  # it is divisible

            index = np.argsort(rg)
            yt = np.squeeze(y_model[:, segment], axis=1)  # swapped : and segment
            zt = np.squeeze(z_model[:, segment], axis=1)
            y_index = np.argsort(yt[index[d - 1]])
            y_sort = np.sort(yt[index[d - 1]])  # swapped : and index[d],

            # estimate where the cut should go
            _, tlen = yt.shape

            if np.fmod(tlen, 2):  # yt has an odd number of elements
                cut = y_sort[int((tlen + 1) / 2)]
            else:  # yt has an even number of elements
                cut = (y_sort[int(tlen / 2)] + y_sort[int(tlen / 2 + 1)]) / 2
                # end of the median calculation

            L = y_sort <= cut

            if np.sum(L) == tlen:  # then the right node will be empty...
                L = y_sort < cut  # ...so use a slightly different boundary
                cut = (cut + np.max(y_sort[L])) / 2
                # end of the cut adjustment

                # adjust the order of the data
            y_model[:, segment] = np.expand_dims(yt[:, y_index], axis=1)
            z_model[:, segment - 1] = zt[:, y_index]

            # mark this as a non-terminal node
            sort_list[node, :] = [index[d - 1], cut]
            node_list[node, 2] = last + 1
            node_list[node, 3] = last + 2
            last = last + 2

            # add the information for the new nodes
            node_list = np.vstack([node_list, [segment[0][0], segment[0][0] + np.sum(L) - 1, 0, 0]])
            node_list = np.vstack([node_list, [segment[0][0] + np.sum(L), segment[0][tlen - 1], 0, 0]])
            sort_list = np.vstack([sort_list, [[0, 0], [0, 0]]])

        # end of the splitting process

        node += 1
        # end of the while loop

    return (y_model, z_model, sort_list, node_list)


def kd_search(node, m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list):
    """
      [] = kdsearch(node)
      node - unknown

    Remarks
    - Search a kd_tree to find the nearest matches to the global variable
      yq, a vector.  The nearest matches will be put in the global variable
      pqr, and their distances in pqd.  See loclin_kd for a usage example.

    - pqd: Starts as an 1x2 with [inf, inf], then it is added to with the values from dist.
      If pqd is longer than m_search, then only the values up to m_search are kept.
    - pqr: Starts as an empty array, then the data from y_model is added with the bounds
      defined by yi which is also defined by node_list. Then the last two data points are
      kept just like with pqd.
    - pqz: The same concept as pqr, except it is using data from the z_model.
    - index: Records the indices that are to be traversed to create the sorted order for pqd.
      This is used later on for pqr & pqz to order those to arrays by the order defined by index.

    Future Work
    - This code could be commmented to be understood easier.

    Feb 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
             - Commented and formated
    """
    if L_done == 1:
        return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

    if node_list[node, 2] == 0:  # it's a terminal node, so...
        # first, compute the distances...
        yi = node_list[node, 0:2]  # index bounds of all y_model to consider
        yt = y_model[:, yi[0]:yi[1]]
        zt = z_model[:, yi[0]:yi[1]]

        d = len(yq)  # get the dimension

        yq1 = yq  # using this to swap the yq array back to a row vector
        yq = yq[:, np.newaxis]  # column vector

        dist = np.sqrt(np.sum((yt[0:d, :] - yq[0:d]) ** 2, axis=0))

        yq = yq1

        # and then sort them and load pqd, pqr, and pqz

        # distances ^2
        pqd = np.append(dist, pqd)
        pqr = np.append(yt, pqr)
        pqz = np.append(zt, pqz)

        index = np.argsort(pqd)  # distances sorted indexes
        pqd = np.sort(pqd)  # distances sorted

        length = pqz.shape[0]

        if len(index) > length:  # to avoid indexing out of bounds
            pqr = pqr[index[0:len(pqz)]]
            pqz = pqz[index[0:len(pqz)]]
        else:  # if index is <= length then we're okay to use the entirety of its length
            pqr = pqr[index]
            pqz = pqz[index]
        # keep only the first m_search points
        if len(pqd) > m_search:
            pqd = pqd[0:m_search]

        length = pqz.shape[0]

        # most of the time this is true
        if length > m_search:
            pqr = pqr[0:m_search]
            pqz = pqz[0:m_search]

        # m_search-1 to not cause indexing issues
        if any((np.abs(yq - b_lower) <= pqd[m_search - 1]) | (np.abs(yq - b_upper) <= pqd[m_search - 1])):
            L_done = 1

        return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)
    else:  # it's not a terminal node, so search a little deeper
        disc = int(sort_list[node, 0])
        part = sort_list[node, 1]

        if yq[disc] <= part:  # determine which child node to go to

            temp = b_upper[disc]
            b_upper[disc] = part
            #         [pqd]=kdsearch(node_list(node,3),m_search,yq,pqd)
            (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list) = kd_search(
                node_list[node, 2], m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list,
                node_list)
            b_upper[disc] = temp
        else:
            temp = b_lower[disc]
            b_lower[disc] = part
            #         [pqd]=kdsearch(node_list(node,4),m_search,yq,pqd)
            (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list) = kd_search(
                node_list[node, 3], m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list,
                node_list)
            b_lower[disc] = temp

        if L_done:
            return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

        if yq[disc] <= part:  # determine whether other child node needs to be searched
            temp = b_lower[disc]
            b_lower[disc] = part

            L = overlap(yq, m_search, pqd, b_upper, b_lower)
            if L == 1:
                #             [pqd]=kdsearch(node_list(node,4),m_search,yq,pqd);
                (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list) = kd_search(
                    node_list[node, 4], m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower,
                    sort_list, node_list);

                b_lower[disc] = temp
        else:
            temp = b_upper[disc]
            b_upper[disc] = part

            L = overlap(yq, m_search, pqd, b_upper, b_lower)
            if L == 1:
                #             [pqd]=kdsearch(node_list(node,3),m_search,yq,pqd);
                (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list) = kd_search(
                    node_list[node, 3], m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower,
                    sort_list, node_list);

                b_upper[disc] = temp

        if L_done:
            return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

    return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)


def overlap(yq, m_search, pqd, b_upper, b_lower):
    """
    L = overlap
    Inputs: yq, m_search, pqd, b_upper, b_lower.
    Outputs: L - unknown

    Remarks
    - References: - "Data Structures for Range Searching", J.L. Bently, J.H.
                    Friedman, ACM Computing Surveys, Vol 11, No 4, p 397-409,
                    December 1979.
                  - "An Algorithm for Finding Best Matches in Logarithmic
                    Expected Time", J.H. Friedman, J.L. Bentley, R.A. Finkel,
                    ACM Transactions on Mathematical Software, Vol 3, No 3,
                    p 209-226, September 1977.

    Future Work
    - None.

    Mar 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
    """
    # indexing with m_search = 2 on a 2 size numpy array causes errors so it is pqd[m_search-1]
    dist = pqd[m_search - 1] ** 2
    sum = 0

    for i in range(len(yq)):

        if yq[i] < b_lower[i]:
            sum = sum + (yq[i] - b_lower[i]) ** 2
            if sum > dist:
                L = 0
                return L
            # end of the sum > dist
        elif yq[i] > b_upper[i]:
            sum = sum + (yq[i] - b_upper[i]) ** 2
            if sum > dist:
                L = 0
                return L
            # end of the sum > dist if
        # end of the yq(i) <> a bound if
    # end of the i loop

    L = 1

    return L


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    x, y, z: Points
    s, r, b: Parameters defining the Lorenz attractor

    x_dot, y_dot, z_dot: Values of the Lorenz attractor's partial derivatives
    at the point x, y, z
    '''

    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z

    return x_dot, y_dot, z_dot


def LyE_R(X, Fs, tau, dim, *args):
    """
      inputs  - X, If this is a single dimentional array the code will use tau
                   and dim to perform a phase space reconstruction. If this is
                   a multidimentional array the phase space reconstruction will
                   not be used.
              - Fs, sampling frequency in units s^-1
              - tau, time lag
              - dim, embedding dimension
      outputs - out, contains the starting matched pairs and the average line
                     divergence from which the slope is calculated. The matched
                     paris are columns 1 and 2. The average line divergence is
                     column 3.
      [LyES,LyEL,out]=LyE_Rosenstein_FC(X,Fs,tau,dim,slope,MeanPeriod,plot)
     inputs  - slope, a four element array with the number of periods to find
                       the regression lines for the short and long LyE. This is
                       converted to indexes in the code.
              - MeanPeriod, used in the slope calculation to find the short and
                            long Lyapunov Exponents.
              - plot, a boolean specifying if a figure should be created
                      displaying the regression lines. This figure is visible
                      by default.
      outputs - LyES, short/local lyapunov exponent
              - LyEL, long/orbital lyapunov exponent
      Remarks
      - This code is based on the algorithm presented by Rosenstein et al,
        1992.
      - Recommendations for the slope input can be found in the references
        below. It is possible a long term exponent can not be found with your
        inputs. If your selection exceeds the length of the data LyEL will
        return as a NaN.
      Future Work
      - It may be possible to sped it up conciderably by re-organizing the for
        loops. A database for the matched points would need to be created.
      References
      - Rosentein, Collins and De Luca; "A practical method for calculating
        largest Lyapunov exponents from small data sets;" 1992
      - Yang and Pai; "Can stability really predict an impending slip-related
        fall among older adults?", 2014
      - Brujin, van Dieen, Meijer, Beek; "Statistical precision and sensitivity
        of measures of dynamic gait stability," 2009
      - Dingwell, Cusumano; "Nonlinear time series analysis of normal and
        pathological human walking," 2000
      Version History
      Jun 2008 - Created by Fabian Cignetti
               - It is suspected this code was originally written by Fabian
                 Cignetti
      Apr 2017 - Revised by Ben Senderling
               - Added comments section. Automated slope calculation. Added
                 calculation of orbital exponent.
      Jun 2020 - Revised by Ben Senderling
               - Incorporated the subroutines directly into the code since they
                 were only used in one location. Converted various for loops
                 into indexed operations. This significantly improved the
                 speed. Added if statements to compensate for errors with the
                 orbital LyE. If the data is such an orbital LyE would not be
                 found with the hardcoded regression line bounds. Made this
                 slope and the file input optional. Removed the MeanPeriod as
                 an imput and made it a calculation in the code. Added the out
                 array so the matched pairs and average line distance can be
                 reviewed, or used to finf the slope. Removed the progress
                 output to the command window since it was sped up
                 conciderably. Edited the figure output. Added code that allows
                 a multivariable input to be entered as X.
      Aug 2020 - Revised by Ben Senderling
               - Removed mean period calculation and turned it into an input.
                 This varies too widely between time series to have it
                 automatically calculated in the script. It was replaced with
                 tau to find paired points.
    """
    # Checked that X is vertically oriented. If X is a single or multiple
    # dimentional array the length is assumed to be longer than the width. It
    # is re-oriented if found to be different.
    X = np.array(X, ndmin=2)
    r, c = np.shape(X)
    if r > c:
        X = np.copy(X.transpose())

    # Checks if a multidimentional array was entered as X.
    if np.size(X, axis=0) > 1:
        M = np.shape(X)[1]
        Y = X
    else:
        # Calculate useful size of data
        N = np.shape(X)[1]
        M = N - (dim - 1) * tau

        Y = np.zeros((M, dim))
        for j in range(dim):
            Y[:, j] = X[:, 0 + j * tau:M + j * tau]
    # Find nearest neighbors

    IND2 = np.zeros((1, M), dtype=int)
    for i in range(M):
        # Find nearest neighbor.
        Yinit = np.matlib.repmat(Y[i], M, 1)
        Ydiff = (Yinit - Y[0:M, :]) ** 2
        Ydisti = np.sqrt(np.sum(Ydiff, axis=1))

        # Exclude points too close based on dominant frequency.
        range_exclude = np.arange(round((i + 1) - tau * 0.8 - 1), round((i + 1) + tau * 0.8))
        range_exclude = range_exclude[(range_exclude >= 0) & (range_exclude < M)]
        Ydisti[range_exclude] = 1e5

        # find minimum distance point for first pair
        IND2[0, i] = np.argsort(Ydisti)[0]

    out = np.vstack((np.arange(M), np.ndarray.flatten(IND2)))

    # Calculate distances between matched pairs.
    DM = np.zeros((M, M))

    IND2len = np.shape(IND2)[1]

    for i in range(IND2len):
        # The data can only be propagated so far from the matched pair.
        EndITL = M - IND2[:, i][0]
        if (M - IND2[:, i][0]) > (M - i):
            EndITL = M - i

        # Finds the distance between the matched paris and their propagated
        # points to the end of the useable data.
        DM[0:EndITL, i] = np.sqrt(
            np.sum((Y[i:EndITL + i, :] - Y[IND2[:, i][0]:EndITL + IND2[:, i][0], :]) ** 2, axis=1))

    # Calculates the average line divergence.
    r, _ = np.shape(DM)

    AveLnDiv = np.zeros(len(DM))
    # NOTE: MATLAB version does not preallocate AveLnDiv, we could preallocate that.
    for i in range(r):
        distanceM = DM[i, :]
        if np.sum(distanceM) != 0:
            AveLnDiv[i] = np.mean(np.log(distanceM[distanceM > 0]))

    out = np.vstack((out, AveLnDiv))

    # Find LyES and LyEL
    plot = 0  # To avoid errors later on


    if len(sys.argv) == 0:
        output_list = out
    else:
        slope = args[0]
        MeanPeriod = args[1]
        plot = args[2]
        output_list = list()

        time = np.arange(0, len(AveLnDiv)) / Fs / MeanPeriod

        shortL = np.zeros(2, dtype=int)
        longL = np.zeros(2, dtype=int)

        # The values in slope are assumed to be the number of periods. These
        # are converted into indexes.
        if slope[0] == 0:
            shortL[0] = 0  # A value of 0 periods cannot be used.
        else:
            shortL[0] = round(slope[0] * MeanPeriod * Fs)

        shortL[1] = round(slope[1] * MeanPeriod * Fs)

        longL[0] = round(slope[2] * MeanPeriod * Fs)
        longL[1] = round(slope[3] * MeanPeriod * Fs)

        # If the index chosen exceeds the length of AveLnDiv then that exponent
        # is made a NaN.
        if shortL[1] <= np.size(np.nonzero(AveLnDiv)):
            slopeinterceptS = poly.polyfit(time[shortL[0]:shortL[1] + 1], AveLnDiv[shortL[0]:shortL[1] + 1], 1)
            LyES = slopeinterceptS[1]
            timeS = time[shortL[0]:shortL[1] + 1]
            LyESline = poly.polyval(timeS, slopeinterceptS)
        else:
            LyES = np.nan

        if longL[1] <= np.size(np.nonzero(AveLnDiv)):
            slopeinterceptL = poly.polyfit(time[longL[0]:longL[1] + 1], AveLnDiv[longL[0]:longL[1] + 1], 1)
            LyEL = slopeinterceptL[1]
            timeL = time[longL[0]:longL[1] + 1]
            LyELline = poly.polyval(timeL, slopeinterceptL)
        else:
            LyEL = np.nan

        output_list.append(LyES)
        output_list.append(LyEL)
        output_list.append(out)

    AveLnDiv = AveLnDiv[np.nonzero(AveLnDiv)]
    time = time[0:len(AveLnDiv)]

    # Plot data

    if plot == 1:
        plt.plot(time, AveLnDiv, color="black")
        plt.title("LyE")
        plt.xlabel("Periods (s)")
        plt.ylabel("<ln(divergence)>")

        if not np.isnan(LyES):
            plt.plot(timeS, LyESline, color="red", linewidth=3, label="LyE_Short = {}".format(LyES))
        if not np.isnan(LyEL):
            plt.plot(timeL, LyELline, color="green", linewidth=3, label="LyE_Long = {}".format(LyEL))

        plt.legend(loc="best")
        plt.show()
    return output_list

def LyE_W(x, Fs, tau, dim, evolve):
    """
    inputs  - x, time series
            - Fs, sampling frequency
            - tau, time lag
            - dim, embedding dimension
            - evolve, parameter of the same name from Wolf's 1985 paper. This
              code expects a number of frames as an input.
    outputs - out, matrix detailing variables at each iteration
            - LyE, largest lyapunov exponent
    [out,LyE] = LyE_W20200820(X,Fs,tau,dim,evolve,SCALEMX,SCALEMN,ANGLMX,ZMULT)
            - SCALEMX, length of which the local structure of the attractor
              is no longer being probed
            - SCALEMN, length below which noise predominates the attractors
              behavior
            - ANGLMX, maximum angle used to constrain replacements
            - ZMULT, multiplier used to increase SCALEMX, unused in the
              current version of the code
    Remarks
    - This code calculates the largest lyapunov exponent of a time series
      according to the algorithm detailed in Wolf's 1985 paper. This code has
      been aligned with his code published on the Matlab file exchange in
      2016. It will largely find the same replacement points, the remaining
      difference being in the replacement algorithm.
    - The varargin can be used to specify some of the secondary parameters in
      the algorithm. All of the extra arguements must be specified if any are
      to be specified at all. Otherwise defaults are used.
    - ZMULT is not currently used in the code but was in a previous version.
      Its place in the subroutine inputs and outputs was kept in case it is
      put back in.
    - It should be noted that the process in the searching algorithm has a
      significant impact on the resulting LyE.
    - The code expects evolve to be the number of frames to use but we
      encourage you to report this as a time-value in publications.
    Prior - Created by Shane Wurdeman, unonbcf@unomaha.edu
          - Adapted by Brian Knarr, unonbcf@unomaha.edu
          - The code previously was influenced heavily by the FORTRAN syntax
            published in Wolf's 1985 paper. These were modified to better
            take advantage of MATLAB and speed up the code.
    Mar 2017 - Modified by Ben Senderling, unonbcf@unomaha.edu
             - Changed parameter "n" to "evolve."
             - Changed "ZMULT" back to 1.
             - Aligned the code with Wolf's Matlab File Exchange submission
               to find the same replacement points. This is now essential his
               algorithm but retains the speed of previous versions.
    Apr 2019 - Modified by Ben Senderling, unonbcf@unomaha.edu
             - Changed line 'range_exclude = range_exclude(range_exclude>=1 &
               range_exclude<=NPT);' to say '>=1' instead of '>1' to prevent
               self matches with the first point. This was indirectly
               accounted for by setting distances less than SCALEMN to 0.
             - '<SCALEMN' was removed from the code entirely and replaced with a
               '<=0'. This was checked against joint angles and EMG data. The
               change did not result in different pairs. This also removes an
              input.
    """

    x = np.array([x])
    SCALEMX = (np.max(x) - np.min(x)) / 10
    ANGLMX = 30 * np.pi / 180
    ZMULT = 1

    DT = 1 / Fs

    ITS = 0
    distSUM = 0

    if np.size(x, axis=0) == 1:
        m = dim
        N = np.size(x, axis=1)
        M = N - (m - 1) * tau
        Y = np.zeros((M, m))

        for i in range(0, m):
            Y[:, i] = x[:, (0 + i * tau):(M + i * tau)]

        NPT = np.size(x, axis=1) - (dim - 1) * tau - evolve  # Size of useable data
        Y = Y[0:NPT + evolve, :]

    else:
        Y = np.array(x)
        NPT = np.size(Y, axis=0) - evolve

    out = np.zeros((int(np.floor(NPT / evolve) + 1), 9), dtype="object")
    thbest = 0
    OUTMX = SCALEMX

    # Find first pair

    # Distance from current point to all other points
    current_point = 0

    Yinit = matlib.repmat(Y[current_point], NPT, 1)
    Ydiff = (Yinit - Y[0:NPT, :]) ** 2
    Ydisti = np.sqrt(np.sum(Ydiff, 1))

    # Exclude points too close on path and close in distance
    range_exclude = np.arange(current_point - 10, current_point + 10 + 1)
    range_exclude = range_exclude[(range_exclude >= 0) & (range_exclude < NPT)]
    Ydisti[Ydisti <= 0] = np.nan
    Ydisti[range_exclude] = np.nan

    # find minimum distance point for first pair
    current_point_pair = np.argsort(Ydisti)[0]

    for i in range(0, NPT, evolve):
        current_point = i
        # calculate starting and evolved distance
        if current_point_pair + evolve < len(Y) and current_point + evolve < len(Y):
            start_dist = np.linalg.norm(Y[current_point, :] - Y[current_point_pair, :])
            end_dist = np.linalg.norm(Y[current_point + evolve, :] - Y[current_point_pair + evolve, :])
        else:
            start_dist = np.linalg.norm(Y[current_point, :] - Y[current_point_pair, :])
            end_dist = np.linalg.norm(Y[current_point + evolve, :] - Y[current_point_pair + evolve - 1, :])

        # calculate total distance so far
        distSUM = distSUM + np.log2(end_dist / start_dist) / (evolve * DT)  # DT is sampling rate?!
        ITS = ITS + 1  # count iterations
        LyE = distSUM / ITS  # max Lyapunov exponent

        #   CPP[i] = current_point_pair # Store found pairs

        out[int(np.floor(i / evolve))] = [ITS, current_point, current_point_pair, start_dist, end_dist, LyE, OUTMX,
                                          (thbest * 180 / np.pi), (ANGLMX * 180 / np.pi)]

        ZMULT = 1

        if end_dist < SCALEMX:
            current_point_pair = current_point_pair + evolve
            if current_point_pair > NPT:
                current_point_pair = current_point_pair - evolve
                flag = 1
                (current_point_pair, ZMULT, ANGLMX, thbest, OUTMX) = get_next_point(flag, Y, current_point,
                                                                                    current_point_pair, NPT, evolve,
                                                                                    SCALEMX, ZMULT, ANGLMX)
            continue
        # find point pairing for next iteration
        flag = 0
        (current_point_pair, ZMULT, ANGLMX, thbest, OUTMX) = get_next_point(flag, Y, current_point, current_point_pair,
                                                                            NPT, evolve, SCALEMX, ZMULT, ANGLMX)

    return (out, LyE)


def get_next_point(flag, Y, current_point, current_point_pair, NPT, evolve, SCALEMX, ZMULT, ANGLMX):
    # Distance from evolved point to all other points
    Yinit = np.matlib.repmat(Y[current_point + evolve, :], NPT, 1)
    Ydiff = (Yinit - Y[0:NPT, :]) ** 2
    Ydisti = np.sqrt(np.sum(Ydiff, axis=1))

    # Exclude points too close on path and close in distance than noise
    range_exclude = np.arange(current_point + evolve - 10, current_point + evolve + 10 + 1)
    range_exclude = range_exclude[(range_exclude >= 0) & (range_exclude < NPT)]
    Ydisti[range_exclude] = np.nan

    if current_point_pair + evolve < len(Y) and current_point + evolve < len(Y):
        end_dist = np.linalg.norm(Y[current_point + evolve, :] - Y[current_point_pair + evolve, :])
    else:
        end_dist = np.linalg.norm(Y[current_point + evolve, :] - Y[current_point_pair + evolve - 1, :])

    # Vector from evolved point to all other points
    Vnew = np.matlib.repmat(Y[current_point + evolve, :], NPT, 1) - Y[:NPT, :]

    # Vector from evolved point to evolved point pair
    if current_point_pair + evolve < len(Y) and current_point + evolve < len(Y):
        PT1 = Y[current_point + evolve, :]
        PT2 = Y[current_point_pair + evolve, :]
    else:
        PT1 = Y[current_point + evolve, :]
        PT2 = Y[current_point_pair + evolve - 1, :]
    Vcurr = PT1 - PT2

    # Angle between evolved pair vector and all other vectors
    # TODO: Had to add a summation here.
    cosTheta = np.abs(np.divide(np.sum(Vcurr.T * Vnew, axis=1), (Ydisti * end_dist)))
    theta = np.arccos(cosTheta)

    # Search for next point
    # -1 Meaning point not found.
    next_point = -1
    while next_point == -1:
        (next_point, ZMULT, ANGLMX, thbest, SCALEMX) = find_next_point(flag, theta, Ydisti, SCALEMX, ZMULT, ANGLMX)

    return next_point, ZMULT, ANGLMX, thbest, SCALEMX


def find_next_point(flag, theta, Ydisti, SCALEMX, ZMULT, ANGLMX):
    # Restrict search based on distance and angle
    PotenDisti = np.copy(Ydisti)
    PotenDisti[(Ydisti <= 0) | (theta >= ANGLMX)] = np.nan

    next_point = -1
    if flag == 0:
        next_point = np.argsort(PotenDisti)[0]
        # if closest angle point is within angle range -> point found and reset
        # search space
        if PotenDisti[next_point] <= SCALEMX:
            ANGLMX = 30 * np.pi / 180
            thbest = np.abs(theta[next_point])
            return (next_point, ZMULT, ANGLMX, thbest, SCALEMX)
        else:
            next_point = -1
            flag = 1
    if flag == 1:
        PotenDisti = np.copy(Ydisti)
        PotenDisti[Ydisti <= 0] = np.nan
        next_point = np.argsort(PotenDisti)[0]
        thbest = ANGLMX

    return (next_point, ZMULT, ANGLMX, thbest, SCALEMX)


def Ent_Samp(data, m, r):
    """
    function SE = Ent_Samp20200723(data,m,r)
    SE = Ent_Samp20200723(data,m,R) Returns the sample entropy value.
    inputs - data, single column time seres
            - m, length of vectors to be compared
            - r, radius for accepting matches (as a proportion of the
              standard deviation)

    output - SE, sample entropy
    Remarks
    - This code finds the sample entropy of a data series using the method
      described by - Richman, J.S., Moorman, J.R., 2000. "Physiological
      time-series analysis using approximate entropy and sample entropy."
      Am. J. Physiol. Heart Circ. Physiol. 278, H2039–H2049.
    - m is generally recommendation as 2
    - R is generally recommendation as 0.2
    May 2016 - Modified by John McCamley, unonbcf@unomaha.edu
             - This is a faster version of the previous code.
    May 2019 - Modified by Will Denton
             - Added code to check version number in relation to a server
               and to automatically update the code.
    Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Removed the code that automatically checks for updates and
               keeps a version history.
    Define r as R times the standard deviation
    """
    R = r * np.std(data)
    N = len(data)

    data = np.array(data)

    dij = np.zeros((N - m, m + 1))
    dj = np.zeros((N - m, 1))
    dj1 = np.zeros((N - m, 1))
    Bm = np.zeros((N - m, 1))
    Am = np.zeros((N - m, 1))

    for i in range(N - m):
        for k in range(m + 1):
            dij[:, k] = np.abs(data[k:N - m + k] - data[i + k])
        dj = np.max(dij[:, 0:m], axis=1)
        dj1 = np.max(dij, axis=1)
        d = np.where(dj <= R)
        d1 = np.where(dj1 <= R)
        nm = d[0].shape[0] - 1  # subtract the self match
        Bm[i] = nm / (N - m)
        nm1 = d1[0].shape[0] - 1  # subtract the self match
        Am[i] = nm1 / (N - m)

    Bmr = np.sum(Bm) / (N - m)
    Amr = np.sum(Am) / (N - m)

    return -np.log(Amr / Bmr)


def Ent_Ap(data, dim, r):
    """
    Ent_Ap20120321
      data : time-series data
      dim : embedded dimension
      r : tolerance (typically 0.2)

      Changes in version 1
          Ver 0 had a minor error in the final step of calculating ApEn
          because it took logarithm after summation of phi's.
          In Ver 1, I restored the definition according to original paper's
          definition, to be consistent with most of the work in the
          literature. Note that this definition won't work for Sample
          Entropy which doesn't count self-matching case, because the count
          can be zero and logarithm can fail.

      *NOTE: This code is faster and gives the same result as ApEn =
             ApEnt(data,m,R) created by John McCamley in June of 2015.
             -Will Denton

    ---------------------------------------------------------------------
    coded by Kijoon Lee,  kjlee@ntu.edu.sg
    Ver 0 : Aug 4th, 2011
    Ver 1 : Mar 21st, 2012
    ---------------------------------------------------------------------
    """

    r = r * np.std(data)
    N = len(data)
    phim = np.zeros(2)
    for j in range(2):
        m = dim + j
        phi = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))
        for i in range(m):
            data_mat[:, i] = data[i:N - m + i + 1]
        for i in range(N - m + 1):
            temp_mat = np.abs(data_mat - data_mat[i, :])
            AorB = np.unique(np.where(temp_mat > r)[0])
            AorB = len(temp_mat) - len(AorB)
            phi[i] = AorB / (N - m + 1)
        phim[j] = np.sum(np.log(phi)) / (N - m + 1)
    AE = phim[0] - phim[1]
    return AE


def RQA(DATA, TYPE, EMB, DEL, ZSCORE, NORM, LINELENGTH, SETPARA, SETVALUE, PLOTOPTION, nargout):
    """
    Usage: (RP, RESULTS)=RQA20210210(DATA,TYPE,EMB,DEL,ZSCORE,NORM,LINELENGTH,SETPARA,SETVALUE,PLOTOPTION)
    Inputs  - DATA, a double-variable with each dimension of the
                    to-be-analyzed signal as a row of numbers in a separate
                    column. If too many columns are present for the TYPE of
                    analysis selected, the other columns will be ignored
                    (i.e. for 'cRQA' only the first two columns will be
                    used).
            - TYPE, a string indicating which type of RQA to run (i.e.
                    'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
                    TYPE = 'RQA'.
            - EMB, the number of embedding dimensions (i.e., EMB = 1 would
                   be no embedding via time-delayed surrogates, just using
                   the provided number of colums as dimensions. The default
                   value is EMB = 1.
            - DEL, the delay parameter used for time-delayed embedding (if
                   EMB > 1). The default value is DEL = 1.
            - ZSCORE, indicates, whether the data (i.e., the different
                   columns of DATA, being the different signals or
                   dimensions of a signal) should be z-scored before
                   performing MdRQA:
                   0 - no z-scoring of DATA
                   1 - z-score columns of DATA
                   The default value is ZSCORE = 0.
            - NORM, the type of norm by with the phase-space is normalized.
                   The following norms are available:
                   'euc' - Euclidean distance norm
                   'max' - Maximum distance norm
                   'min' - Minimum distance norm
                   'non' - no normalization of phase-space
                   The default value is NORM = 'non'.
            - SETPARA, the parameter which you would like to set a target
                   value for the recurrence plot (i.e. 'radius' or
                   'recurrence'). The default value is SETPARA = 'radius'.
            - SETVALUE, sets the value of the selected parameter. If
                   SETVALUE = 1, then the radius will be set to 1 if SETPARA
                   = 'radius' or the radius will be adjusted until the
                   recurrence is equal to 1 if SETPARA = 'recurrence'. The
                   default value if SETPARA = 'radius' is 1. The default
                   value if SETPARA = 'recurrence' is 2.5.
            - PLOTOPTION, a 1 will display a plot, 0 will not display it.
            - nargout, a number of arguments we want to return from this function.
    Outputs - RP is a matrix holding the resulting recurrence plot.
            - RESULTS is a dictionary holding the following recurrence
              variables:
              1.  DIM    - dimension of the input data (used for mdRQA)
              2.  EMB    - embedding dimension used in the calculation of the
                           distance matrix
              3.  DEL    - time lag used in the calculation of the distance
                           matrix
              4.  RADIUS - radius used for the recurrence plot
              5.  NORM   - type of normilization used for the distance matrix
              6.  ZSCORE - whether or not zscore was used
              7.  Size   - size of the recurrence plot
              8.  %REC   - percentage of recurrent points
              9.  %DET   - percentage of diagonally adjacent recurrent points
              10. MeanL  - average length of adjacent recurrent points
              11. MaxL   - maximum length of diagonally adjacent recurrent
                           points
              12. EntrL  - Shannon entropy of distribution of diagonal lines
              13. %LAM   - percentage of vertically adjacent recurrent points
              14. MeanV  - average length of diagonally adjacent recurrent
                           points
              15. MaxV   - maximum length of vertically adjacent recurrent
                           points
              16. EntrV  - Shannon entropy of distribution of vertical lines
              17. EntrW  - Weighted entropy of distribution of vertical
                           weighted sums
    Remarks
    - Computes a recurrence plot for either recurrence quantification
      analysis (RQA), cross recurrence quantification analysis (cRQA), joint
      recurrence quantification analysis (jRQA), or multidimensional
      recurrence quantification analysis (mdRQA). Either radius or target
      recurrence can be set.
    Reference:
    - Wallot, S., Roepstorff, A., & Monster, D. (2016). Multidimensional
      Recurrence Quantification Analysis (MdRQA) for the analysis of
      multidimensional time-series: A software implementation in MATLAB and
      its application to group-level data in joint action. Frontiers in
      Psychology, 7, 1835. http://dx.doi.org/10.3389/fpsyg.2016.01835
    - Eroglu, D., Peron, T. K. D., Marwan, N., Rodrigues, F. A., Costa, L. D.
      F., Sebek, M., ... & Kurths, J. (2014). Entropy of weighted recurrence
      plots. Physical Review E, 90(4), 042919.

    Jul 2016 Modified by Sebastian Wallot
             - VERSION 1.0.0
               28. July 2016 by Sebastian Wallot, Max Planck Insitute for
               Empirical Aesthetics, Frankfurt, Germany & Dan M?nster, Aarhus
               University, Aarhus, Denmark
    Jul 2017 Modified by Will Denton
             - VERSION 1.1.0
               06. July 2017 by Will Denton (wdenton@unomaha.edu), Troy Rand
               (troyrand@gmail.com), and Casey Wiens (cwiens32@gmail.com),
               Biomechanics Research Building, University of Nebraska at
               Omaha. Changes include cleaning up some errors, making the
               default input arguments function correctly, incorperating
               other types of RQA (e.g. RQA, CRQA, JRQA), allowing %REC to be
               set instead of radius, incorporating weighted recurrence
               plots, and adding weighted entropy.
    May 2019 Modified by Will Denton
             - VERSION 1.1.1 (05/09/2019)
             - Added updates/patching.
             - Added usage tracking to allow the Department of Biomechanics
               at the University of Omaha see which codes and versions are
               being used.
             - Added error reporting to allow the Department of Biomechanics
               at the University of Omaha to make improvements to this code.
    Nov 2019 Modified by Will Denton
             - VERSION 1.1.2 (11/18/2019)
             - Fixed usage and error reporting to work with MacOS.
             - Fixed left plot to align with the recurrence plot when zoomed
               in and panning around.
    Jul 2020 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Removed automatic update code and version history code.
    Dec 2020 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Commented out waitbar and added PLOTOPTION input to control
               figure creation.
    Feb 2021 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Made line length an input with a default of 1.
    Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    # Set default parameters if no input exists
    # If SETPARA is not specified, set to 'radius'

    DATA = np.array(DATA)

    if SETPARA == None:
        SETPARA = 'radius'

    # If SETVALUE is not specified, set to 1 if radius is set or 2.5 if perRec is set

    if SETVALUE == None:
        if SETPARA == 'radius' or SETPARA == 'rad' or SETPARA == 1:
            radius = 1
            runSetRad = 0
        elif SETPARA == 'perrec' or SETPARA == 'recurrence' or SETPARA == 2:
            radiusStart = 0.01
            radiusEnd = 0.5
            runSetRad = 1
            SETVALUE = 2.5
    else:
        if SETPARA == 'radius' or SETPARA == 'rad' or SETPARA == 1:
            radius = SETVALUE
            runSetRad = 0
        elif SETPARA == 'perrec' or SETPARA == 'recurrence' or SETPARA == 2:
            radiusStart = 0.01
            radiusEnd = 0.5
            runSetRad = 1

    # If LINELENGTH is not specified, set to '1'
    if LINELENGTH == None:
        LINELENGTH = 1
    # If NORM is not specified, set to 'non'
    if NORM == None:
        NORM = 'non'
    # If ZSCORE is not specified, set to 0
    if ZSCORE == None:
        ZSCORE = 0
    # If DEL is not specified, set to 1
    if DEL == None:
        DEL = 1
    # If EMB is not specified, set to 1
    if EMB == None:
        EMB = 1
    # If EMB is not specified, set to 1
    if TYPE == None:
        TYPE = 'RQA'
    # If z score is selected then z score the data
    if ZSCORE:  # == 1
        DATA = st.zscore(DATA)

    # Set DIM and select proper column(s) of data if too many exist
    # if 1 dimensional, only one column, else column is reflected by the second value of the tuple from shape.
    if DATA.ndim == 1:
        r = 1
    else:
        r = np.shape(DATA)[0]

    TYPE = TYPE.upper()

    if TYPE == 'RQA':
        TYPE = 'RQA'
    elif TYPE == 'CRQA' or TYPE == 'CROSS':
        TYPE = 'CRQA'
    elif TYPE == 'JRQA' or TYPE == 'JOINT':
        TYPE = 'JRQA'
    elif TYPE == 'MDRQA' or TYPE == 'MD' or TYPE == 'MULTI':
        TYPE = 'MDRQA'

    if TYPE == 'RQA':
        DIM = 1
        if r > 1:
            DATA = DATA[:, 0]
            warnings.warn("More than one column of data. Only using first column.")
    elif TYPE == 'CRQA':
        DIM = 2
        if r > 2:
            DATA = DATA[:, :2]
            warnings.warn("More than two columns of data Only using first two columns.")
    elif TYPE == 'JRQA':
        DIM = r
        if r < 2:
            raise Exception("Input data must have at least two columns.")
    elif TYPE == 'MDRQA':
        DIM = r

    # Embed the data
    if EMB > 1:
        # Preallocation is to be implemented.
        # if r > 1:
        #     tempDATA = np.zeros((DIM*EMB,DATA.shape[1]-(EMB-1)*DEL))
        # else:
        #     tempDATA = np.zeros((DIM*EMB,DATA.shape[0]-(EMB-1)*DEL))
        onerow = False
        for i in range(EMB):
            if i == 0:
                try:
                    tempDATA = np.array(DATA[:, i * DEL:DATA.shape[1] - (EMB - i) * DEL + 1])
                except:
                    onerow = True
                    tempDATA = np.expand_dims(np.array(DATA[i * DEL:DATA.shape[0] - (EMB - i) * DEL + 1]), axis=0)
            elif onerow:
                tempDATA = np.concatenate((tempDATA, np.array([DATA[i * DEL:DATA.shape[0] - (EMB - i) * DEL + 1]])),
                                          axis=0)
            else:
                tempDATA = np.concatenate((tempDATA, DATA[:, i * DEL:DATA.shape[1] - (EMB - i) * DEL + 1]), axis=0)

        DATA = tempDATA
        tempDATA = np.delete(tempDATA, np.s_[::])  # deletes everything from array.

    a = [i for i in range(r)]
    if TYPE == 'RQA':
        if EMB > 1:
            pairs = DATA.T.copy()
            a[0] = pairwise_distances(pairs, metric='euclidean')
            a[0] = np.abs(a[0]) * -1  # make values negative
        else:
            a[0] = pairwise_distances(DATA.reshape(-1, 1), metric='euclidean')
            a[0] = np.abs(a[0]) * -1  # make values negative
    elif TYPE == 'CRQA':
        # Euclidean distance between the first and second column
        if EMB > 1:
            # indexes our pairs from the rows of the data by steps of DIM (for when EMB > 1)
            pairs1 = DATA[np.s_[::DIM], :].T.copy()
            pairs2 = DATA[np.s_[1::DIM], :].T.copy()

            a[0] = pairwise_distances(pairs1, pairs2, metric='euclidean')
            a[0] = np.abs(a[0]) * -1
        else:
            a[0] = pairwise_distances(DATA[0].reshape(-1, 1), DATA[1].reshape(-1, 1), metric='euclidean')
            a[0] = np.abs(a[0]) * -1
        # doing this to avoid unneccesarily creating a weighted recurrence plot later on.
        a = [a[0]]
    elif TYPE == 'JRQA':
        for i in range(r):
            if EMB > 1:
                pairs = DATA[np.s_[i::DIM], :].T.copy()
                a[i] = pairwise_distances(pairs, metric='euclidean')
                a[i] = np.abs(a[i]) * -1
            else:
                a[i] = pairwise_distances(DATA[i, :].reshape(-1, 1), DATA[i, :].reshape(-1, 1), metric='euclidean')
                a[i] = np.abs(a[i]) * -1
    elif TYPE == 'MDRQA':
        a[0] = pairwise_distances(DATA.T, metric='euclidean')
        a[0] = np.abs(a[0]) * -1
        a = [a[0]]

    # Normalize distance matrix
    if 'euc' in NORM:
        for i in range(len(a)):
            b = np.mean(a[i][np.where(a[i] < 0)])
            b = np.negative(np.sqrt(np.abs((b ** 2) + 2 * (DIM * EMB))))
            a[i] = a[i] / np.abs(b)
    elif 'min' in NORM:
        for i in range(len(a)):
            b = np.max(a[i][np.where(a[i] < 0)])
            a[i] = a[i] / np.abs(b)
    elif 'max' in NORM:
        for i in range(len(a)):
            b = np.min(a[i][np.where(a[i] < 0)])
            a[i] = a[i] / abs(b)
    elif 'non' in NORM:
        pass  # do nothing
    else:
        raise Exception('No appropriate norm parameter specified.')

    # Create weighted recurrence plot
    index = -1
    for i in range(len(a) - 1):
        a[i + 1] = np.multiply(a[i], a[i + 1])
        index = i
    if index == 0:
        a = np.negative((np.abs(a[index + 1])) ** (1 / (index + 2)))

    if TYPE in 'RQA' or TYPE in 'CRQA' or TYPE in 'MDRQA':
        a = a[0]

    # Calculate recurrence plot
    if SETPARA == 'radius' or SETPARA == 'rad' or SETPARA == 1:
        perRec = rqaPerRec(a, TYPE, radius, LINELENGTH)
        (diag_hist, vertical_hist) = rqaHistograms(a.copy(), DATA, TYPE, radius)
    elif SETPARA == 'perrec' or SETPARA == 'recurrence' or SETPARA == 2:
        (perRec, radius) = setRadius(a, TYPE, radiusStart, SETVALUE, radiusEnd, LINELENGTH)
        (diag_hist, vertical_hist) = rqaHistograms(a.copy(), DATA, TYPE, radius)

    RESULTS = {
        "DIM": DIM,
        "EMB": EMB,
        "DEL": DEL,
        "RADIUS": radius,
        "NORM": NORM,
        "ZSCORE": ZSCORE,
        "SIZE": len(a),
        "REC": perRec
    }
    if RESULTS["REC"] > 0:
        RESULTS["DET"] = 100 * np.sum(diag_hist[diag_hist > LINELENGTH]) / np.sum(diag_hist)
        RESULTS["MeanL"] = np.mean(diag_hist[diag_hist > LINELENGTH])
        RESULTS["MaxL"] = np.max(diag_hist)
        # Create our histogram using data points that are greater than LINELENGTH (Default is 1 for LINELENGTH)
        # Our bins to use is a range from the minimum to the maximum
        (count, bins) = np.histogram(diag_hist[diag_hist > LINELENGTH])
        total = np.sum(count)
        p = np.divide(count, total)
        zero_indices = np.where(count == 0)
        p = np.delete(p, zero_indices)
        RESULTS["EntrL"] = np.negative(np.sum(np.multiply(p, np.log2(p))))
        RESULTS["LAM"] = 100 * np.sum(vertical_hist[vertical_hist > LINELENGTH]) / np.sum(vertical_hist)
        RESULTS["MeanV"] = np.mean(vertical_hist[vertical_hist > LINELENGTH])
        RESULTS["MaxV"] = np.max(vertical_hist)
        (count, bins) = np.histogram(vertical_hist[vertical_hist > LINELENGTH])
        total = np.sum(count)
        p = np.divide(count, total)
        zero_indices = np.where(count == 0)
        p = np.delete(p, zero_indices)
        RESULTS["EntrV"] = np.negative(np.sum(np.multiply(p, np.log2(p))))
        RESULTS["EntrW"] = RQA_WeightedEntropy(a)
    else:
        RESULTS["DET"] = np.nan
        RESULTS["MeanL"] = np.nan
        RESULTS["MaxL"] = np.nan
        RESULTS["EntrL"] = np.nan
        RESULTS["LAM"] = np.nan
        RESULTS["MeanV"] = np.nan
        RESULTS["MaxV"] = np.nan
        RESULTS["EntrV"] = np.nan
        RESULTS["EntrW"] = np.nan

    # a[a >= np.negative(radius)] = 1.
    # a[a < np.negative(radius)] = 0.

    RP = rotate(1. - a, 90)

    if nargout == 1:
        out = [RESULTS]
    elif nargout == 2:
        out = [RESULTS, RP]

    if PLOTOPTION > 0:

        title = "DIM = {}, EMB = {}, DEL = {}, RAD = {:.5f}, NORM = {}, ZSCORE = {}".format(DIM, EMB, DEL, radius, NORM,
                                                                                            ZSCORE)
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(5, 5, hspace=.7, wspace=.7)
        main_plot = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], xticklabels=[], aspect='auto')
        x_plot = fig.add_subplot(grid[-1, 1:], yticklabels=[], aspect='auto')
        main_plot.imshow(RP, cmap='gray')
        main_plot.set_title(title, fontsize='x-small')
        main_plot.set_xticks([])
        main_plot.set_yticks([])
        main_plot.set_xlabel('X(i)', fontsize='large')
        main_plot.set_ylabel('Y(j)', fontsize='large')
        x_plot.set_xmargin(0)
        x_plot.set_ymargin(0)
        y_plot.set_xmargin(0)
        y_plot.set_ymargin(0)
        if TYPE in 'RQA' or TYPE in 'MDRQA':
            if DATA.ndim == 1: DATA = np.expand_dims(DATA, axis=0)
            x_plot.plot(np.arange(DATA.shape[1]), DATA[0], '-k')
            y_plot.plot(np.flip(DATA[0]), np.arange(DATA.shape[1], 0, -1), '-k')
        elif TYPE in 'CRQA':
            x_plot.plot(np.arange(DATA.shape[1]), DATA[0, :], 'k-')
            y_plot.plot(np.flip(DATA[1, :]), np.arange(DATA.shape[1], 0, -1), '-k')  # HACK: Might be plotting backward.
        elif TYPE in 'JRQA':
            # TODO: MATLAB side shows only one line along the x-axis, our plot ends up plotting more than one line... which one is correct?
            y_plot.plot(np.flip(DATA[0, :]), np.arange(DATA.shape[1]), '-k')
            y_plot.invert_yaxis()
            for i in range(r):
                x_plot.plot(np.arange(DATA.shape[1]), DATA[i, :], 'k-')
        fig.text(.02, .24, "%REC = {:.2f}".format(RESULTS["REC"]))
        fig.text(.02, .21, "%DET = {:.2f}".format(RESULTS["DET"]))
        fig.text(.02, .18, "MaxL = {:.0f}".format(RESULTS["MaxL"]))
        fig.text(.02, .15, "MeanL = {:.2f}".format(RESULTS["MeanL"]))
        fig.text(.02, .12, "EntrL = {:.2f}".format(RESULTS["EntrL"]))
        fig.text(.02, .09, "%LAM = {:.2f}".format(RESULTS["LAM"]))
        fig.text(.02, .06, "MaxV = {:.0f}".format(RESULTS["MaxV"]))
        fig.text(.02, .03, "MeanV = {:.2f}".format(RESULTS["MeanV"]))
        fig.text(.02, .0, "EntrV = {:.2f}".format(RESULTS["EntrV"]))
        if PLOTOPTION == 1:
            plt.show()
    # TODO: Tabs are something that matplotlib does not do by default, if needed this tab feature may be implemented later.
    if PLOTOPTION > 1:
        title = "DIM = {}, EMB = {}, DEL = {}, RAD = {:.5f}, NORM = {}, ZSCORE = {}".format(DIM, EMB, DEL, radius, NORM,
                                                                                            ZSCORE)
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(5, 5, hspace=.7, wspace=.7)
        main_plot = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], xticklabels=[], aspect='auto')
        x_plot = fig.add_subplot(grid[-1, 1:], yticklabels=[], aspect='auto')
        main_plot.imshow(RP, cmap='hot')
        main_plot.set_title(title, fontsize='x-small')
        main_plot.set_xticks([])
        main_plot.set_yticks([])
        main_plot.set_xlabel('X(i)', fontsize='large')
        main_plot.set_ylabel('Y(j)', fontsize='large')
        x_plot.set_xmargin(0)
        x_plot.set_ymargin(0)
        y_plot.set_xmargin(0)
        y_plot.set_ymargin(0)
        if TYPE in 'RQA' or TYPE in 'MDRQA':
            if DATA.ndim == 1: DATA = np.expand_dims(DATA, axis=0)
            x_plot.plot(np.arange(DATA.shape[1]), DATA[0], '-k')
            y_plot.plot(np.flip(DATA[0]), np.arange(DATA.shape[1], 0, -1), '-k')
        elif TYPE in 'CRQA':
            x_plot.plot(np.arange(DATA.shape[1]), DATA[0, :], 'k-')
            y_plot.plot(np.flip(DATA[1, :]), np.arange(DATA.shape[1], 0, -1), '-k')  # HACK: Might be plotting backward.
        elif TYPE in 'JRQA':
            # TODO: MATLAB side shows only one line along the x-axis, our plot ends up plotting more than one line... which one is correct?
            y_plot.plot(np.flip(DATA[0, :]), np.arange(DATA.shape[1]), '-k')
            y_plot.invert_yaxis()
            for i in range(r):
                x_plot.plot(np.arange(DATA.shape[1]), DATA[i, :], 'k-')
        fig.text(.02, .24, "EntrW = {:.2f}".format(RESULTS["EntrW"]))
        plt.show()
    return out


# Function for setting radius to achieve a certain recurrence
def setRadius(a, TYPE, radius, SETVALUE, radiusEnd, LINELENGTH):
    """
    Usage: (perRec, radiusFinal) = setRadius(a, TYPE, radius, SETVALUE, radiusEnd, LINELENGTH)
    Input:  - a, pairwise euclidean distance values of our DATA matrix.
            - TYPE, a string indicating which type of RQA to run (i.e.
                'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
                TYPE = 'RQA'.
            - radius,
            - SETVALUE,sets the value of the selected parameter. If
                SETVALUE = 1, then the radius will be set to 1 if SETPARA
                = 'radius' or the radius will be adjusted until the
                recurrence is equal to 1 if SETPARA = 'recurrence'. The
                default value if SETPARA = 'radius' is 1. The default
                value if SETPARA = 'recurrence' is 2.5.
            - radiusEnd,
            - LINELENGTH,
    Output: - perRec,
            - radiusFinal,
    """
    # Find the radius to provide target # recurrence
    perRec = rqaPerRec(a.copy(), TYPE, radius, LINELENGTH)
    while perRec == 0 or perRec > 2.5:
        # if radius is too small
        #     print('Minimum radius has been adjusted...')
        #     radiusEnd = radius + 0.5;
        if perRec == 0:
            radius = radius * 2
        elif perRec > SETVALUE:
            radius = radius / 1.5
            #                 radiusEnd =  radius + 0.5
        perRec = rqaPerRec(a.copy(), TYPE, radius, LINELENGTH)

    perRec = rqaPerRec(a.copy(), TYPE, radiusEnd, LINELENGTH)
    while perRec < SETVALUE:
        # if radiusEnd is too large
        # print('Maximum radius has been increased...')
        radiusEnd = radiusEnd * 2
        perRec = rqaPerRec(a.copy(), TYPE, radiusEnd, LINELENGTH)

    # Search for radius with target # recurrence

    target = SETVALUE  # designate what percent recurrence is wanted
    iterations = 20  # Number of iterations to find radius
    lv = np.zeros(iterations + 1)  # +1 to hold initial low value
    hv = np.zeros(iterations + 1)  # +1 to hold initial high value
    lv[0] = radius  # set low value
    hv[0] = radiusEnd  # set high value
    perRecIter = np.zeros(iterations)
    mid = np.zeros(iterations)
    rad = np.zeros(iterations)
    # TODO: This converges to a different value than the MATLAB code.
    for i1 in range(iterations):
        mid[i1] = (lv[i1] + hv[i1]) / 2  # find midpoint between hv and lv
        rad[i1] = mid[i1]  # new radius for this iteration
        # Compute recurrence matrix
        perRec = rqaPerRec(a.copy(), TYPE, rad[i1], LINELENGTH)

        perRecIter[i1] = perRec  # set percent recurrence

        if perRecIter[i1] < target:
            # if percent recurrence is below target percent recurrence
            hv[i1 + 1] = hv[i1]
            lv[i1 + 1] = mid[i1]
        else:
            # if percent recurrence is above or equal to target percent recurrence
            lv[i1 + 1] = lv[i1]
            hv[i1 + 1] = mid[i1]

    perRecFinal = perRecIter[-1]  # set final percent recurrence
    radiusFinal = rad[-1]  # set radius for final percent recurrence

    return perRecFinal, radiusFinal


def rqaPerRec(a, TYPE, radius, LINELENGTH):
    """
    Usage: perRec = rqaPerRec(A, TYPE, radius, LINELENGTH)
    Input:  - a, pairwise euclidean distance values from our DATA matrix.
            - TYPE, a string indicating which type of RQA to run (i.e.
               'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
               TYPE = 'RQA'.
            - radius,
            - LINELENGTH
    Output: - perRec
    """

    if not isinstance(a, list):
        a = [a]
    for i2 in range(len(a)):
        nradius = np.negative(radius)
        a[i2][np.where(a[i2] >= nradius)] = 1
        a[i2][np.where(a[i2] < nradius)] = 0
    if len(a) > LINELENGTH:
        for i3 in range(len(a) - 1):
            a[i3 + 1] = np.multiply(a[i3], a[i3 + 1])
        a = a[i3 + 1]
    else:
        a = a[0]

    # Calculate percent recurrence
    if 'CRQA' not in TYPE:
        perRec = 100 * (np.sum(a) - len(a)) / (len(a) ** 2 - len(a))
    else:
        perRec = 100 * (np.sum(a)) / (len(a) ** 2)

    return perRec


def rqaHistograms(a, DATA, TYPE, radius):
    """
    Usage: (diag_hist, vertical_hist) = rqaHistograms(A, DATA, TYPE, radius)
    Input:  - a, stores the recurrence matrix.
            - DATA,
            - TYPE,
            - radius
    Output: - diag_hist,
            - vertical_hist,
    """
    if not isinstance(a, list):
        a = [a]

    for i2 in range(len(a)):
        nradius = np.negative(radius)
        a[i2][np.where(a[i2] >= nradius)] = 1
        a[i2][np.where(a[i2] < nradius)] = 0

    if len(a) > 1:
        for i3 in range(len(a) - 1):
            a[i3 + 1] = np.multiply(a[i3], a[i3 + 1])
        a = a[i3 + 1]
    else:
        a = a[0]

    # If one dimensional, expand to two dimensional for the work up ahead.
    if DATA.ndim == 1:
        DATA = np.expand_dims(DATA, axis=0)

    diag_hist = np.array([])
    vertical_hist = np.array([])
    for i4 in range(-DATA.shape[1], DATA.shape[1]):  # caluculate diagonal line distribution
        diagonal = np.diag(a.copy(), k=i4).astype(int)
        d = label_components(diagonal)
        # lengths
        d = d[np.nonzero(d)]
        d = np.bincount(d)[1:]

        diag_hist = np.append(diag_hist, d)

    # This removes the line of identity in RQA, jRQA, and mdRQA
    if 'CRQA' not in TYPE:
        diag_hist = diag_hist[diag_hist < np.max(diag_hist)]
        if len(diag_hist) == 0:
            diag_hist = 0

    for i5 in range(DATA.shape[1]):
        C = a[:, i5].copy().astype(int)
        v = label_components(C)

        v = v[np.nonzero(v)]
        v = np.bincount(v)[1:]

        vertical_hist = np.append(vertical_hist, v)

    return (diag_hist, vertical_hist)


# Calculate entropy of weighted recurrence plot
def RQA_WeightedEntropy(WRP):
    """
    Usage: Swrp = RQA_WeightedEntropy(WRP)
    Input:  WRP,
    Output: Swrp,
    """
    N = len(WRP)
    si = np.zeros(N)
    for j in range(N):
        si[j] = np.sum(WRP[:, j])
    mi = min(si)
    ma = max(si)
    m = (ma - mi) / 49
    I = 1
    S = np.sum(si)
    p1 = np.array([])
    step = m

    # append to p1 the initial value
    P = np.sum(si[(si >= mi) & (si < (mi + step))])
    p1 = np.append(p1, P / S)
    m = m + mi
    while m < ma:
        # sum of values within the range between m and m+step
        P = np.sum(si[(si >= m) & (si < (m + step))])
        p1 = np.append(p1, P / S)
        m += step
    pp = np.zeros(len(p1))
    for i in range(len(p1)):
        pp[i] = p1[i] * np.log(p1[i])

    pp[np.isnan(pp)] = 0

    Swrp = -1 * (np.sum(pp))
    return Swrp


def label_components(bimg):
    """
    Input:  - bimg: The binary column to process
    Output: - L: The labeled array.
    Remarks:
        This subfunction serves to model the bwlabel function that exists in the MATLAB image processing library.
        Link: https://www.mathworks.com/help/images/ref/bwlabel.html

        The only exception is that this subfunction only handles one dimensional input, we are only expecting
        similar input from what you see in the RQA subfunction rqaHistograms. Our input consists of a NumPy
        array that may look something like: [0 1 1 1 0 0 0 1 1 0 1], our output would look something like
        [0 1 1 1 0 0 0 2 2 0 3], which gives each 'component' it's own label starting from 1 and incrementing
        for each time we encounter a 1 after encountering 0.

        The function in MATLAB takes in a secondary function which outlines connectivity, due to the input being
        one dimensional, we're ignoring that.
    """
    bimg.setflags(write=1)
    label_value = 0
    in_component = False  # using a boolean so we are not incrementing while we are within a certain 'component'
    for i in range(bimg.shape[0]):
        if bimg[i] == 1 and not in_component:
            # mark that we are in a component, increment component value, assign that value to our 1s.
            in_component = True
            label_value += 1
            bimg[i] = label_value
        elif bimg[i] == 1 and in_component:
            # while we are in a component, just mark the values with the component value
            bimg[i] = label_value
        else:
            # if we encounter a 0, then we are not in_component.
            in_component = False
    return bimg
# Set other parameters
# dt = 0.01
# num_steps = 10000
#
# # Initial values require one or more
# xs = np.empty(num_steps + 1)
# ys = np.empty(num_steps + 1)
# zs = np.empty(num_steps + 1)
#
# # Initial values setting
# xs[0], ys[0], zs[0] = (0., 1., 1.05)
#
# # Step through "time", calculating the partial derivatives at the current point
# # and estimate the next point
# for i in range(num_steps):
#     x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
#     xs[i + 1] = xs[i] + (x_dot * dt)
#     ys[i + 1] = ys[i] + (y_dot * dt)
#     zs[i + 1] = zs[i] + (z_dot * dt)
#
# Time_delay = Time_delay(xs, 20, "Chaotic", 0)
# d = Culculation_of_embending_dimensions(xs, 16, 40, 0, "Vaggelis")

