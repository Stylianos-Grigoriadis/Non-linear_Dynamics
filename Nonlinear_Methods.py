import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy.matlib as matlib
import sys

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

