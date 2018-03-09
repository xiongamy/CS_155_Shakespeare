########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding.s If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import math

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''
        
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(1, M + 1):
            # get current token
            tok = x[i - 1]
            if i == 1:
                # first column, take from A_start
                for j in range(self.L):
                    p = self.A_start[j] * self.O[j][tok]
                    probs[1][j] = -float('inf') if p == 0 else math.log(p)
                    seqs[1][j] = str(j)
            else:
                # other columns, take from A
                prev = probs[i - 1]
                first = True
                for prev_j in range(self.L):
                    for j in range(self.L):
                        pp = prev[prev_j]
                        p = self.A[prev_j][j] * self.O[j][tok]
                        log_p = -float('inf') if p == 0 else pp + math.log(p)
                        if first or log_p > probs[i][j]:
                            probs[i][j] = log_p
                            seqs[i][j] = seqs[i - 1][prev_j] + str(j)
                    first = False

        max_seq = ''
        max_log_p = -float('inf')
        for j in range(self.L):
            if probs[M][j] > max_log_p:
                max_seq = seqs[M][j]
                max_log_p = probs[M][j]
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(1, M + 1):
            tok = x[i - 1]
            if i == 1:
                for z in range(self.L):
                    alphas[1][z] = self.O[z][tok] * self.A_start[z]
            else:
                prev = alphas[i - 1]
                for z in range(self.L):
                    for j in range(self.L):
                        alphas[i][z] += self.O[z][tok] * prev[j] * self.A[j][z]
    
        if normalize:
            for i in range(M + 1):
                sum = 0
                for j in range(L):
                    sum += alphas[i][j]
                if sum != 0:
                    for j in range(L):
                        alphas[i][j] /= sum
            
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        
        for i in range(M, 0, -1):
            if i == M:
                for z in range(self.L):
                    betas[M][z] = 1.
            else:
                tok = x[i]
                prev = betas[i + 1]
                for z in range(self.L):
                    for j in range(self.L):
                        betas[i][z] += prev[j] * self.A[z][j] * self.O[j][tok]

        if normalize:
            for i in range(M + 1):
                sum = 0
                for j in range(L):
                    sum += betas[i][j]
                if sum != 0:
                    for j in range(L):
                        betas[i][j] /= sum
                    
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.
        t = [0. for _ in range(self.L)]
        self.A = [[0. for _ in range(self.L)] for _ in range(self.L)]
        for y_list in Y:
            for i in range(len(y_list) - 1):
                y = y_list[i]
                yn = y_list[i + 1]
                self.A[y][yn] += 1
                t[y] += 1
        
        for i in range(self.L):
            for j in range(self.L):
                self.A[i][j] /= t[i]

        # Calculate each element of O using the M-step formulas.

        N = len(X)
        t = [0. for _ in range(self.L)]
        self.O = [[0. for _ in range(self.D)] for _ in range(self.L)]
        for li in range(N):
            x_list = X[li]
            y_list = Y[li]
            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                self.O[y][x] += 1
                t[y] += 1
        
        for i in range(self.L):
            for j in range(self.D):
                self.O[i][j] /= t[i]


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        N = len(X)
        
        for _ in range(N_iters):
            Ps = []
            Pp = []
            
            # Expectation
            for x in X:
                M = len(x)
                alphas = self.forward(x)
                betas = self.backward(x)
                
                # P_single: first index is 1 more than the y value being predicted
                #           second index is the value of the y being predicted
                P_single = [[0. for _ in range(self.L)] for _ in range(M + 1)]
                for i in range(1, M + 1):
                    sum = 0.
                    for z in range(self.L):
                        prod = alphas[i][z] * betas[i][z]
                        P_single[i][z] = prod
                        sum += prod
                    if sum != 0.:
                        for z in range(self.L):
                            P_single[i][z] /= sum
                Ps.append(P_single)
                
                # P_pair: first index is the next y index being predicted
                #         second index is the value of the prev y
                #         third index is the value of the next y
                P_pair = [[[0. for _ in range(self.L)] for _ in range(self.L)] for _ in range(M + 1)]
                for i in range(2, M + 1):
                    sum = 0.
                    for a in range(self.L):
                        for b in range(self.L):
                            prod = alphas[i - 1][a] * self.A[a][b] * self.O[b][x[i - 1]] * betas[i][b]
                            P_pair[i][a][b] = prod
                            sum += prod
                    if sum != 0.:
                        for a in range(self.L):
                            for b in range(self.L):
                                P_pair[i][a][b] /= sum
                Pp.append(P_pair)
                
            # Maximization
            # Calculate each element of A using the M-step formulas.
            self.A = [[0. for _ in range(self.L)] for _ in range(self.L)]
            for a in range(self.L):
                for b in range(self.L):
                    sum = 0.
                    for j in range(N):
                        M = len(X[j])
                        for i in range(2, M + 1):
                            self.A[b][a] += Pp[j][i][b][a]
                            sum += Ps[j][i - 1][b]
                    self.A[b][a] /= sum

            # Calculate each element of O using the M-step formulas.
            self.O = [[0. for _ in range(self.D)] for _ in range(self.L)]
            for z in range(self.L):
                for w in range(self.D):
                    sum = 0.
                    for j in range(N):
                        x = X[j]
                        M = len(x)
                        for i in range(1, M + 1):
                            if w == x[i - 1]:
                                self.O[z][w] += Ps[j][i][z]
                            sum += Ps[j][i][z]
                    self.O[z][w] /= sum

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        for i in range(M):
            if i == 0:
                r = random.random()
                for j in range(self.L):
                    if r < self.A_start[j]:
                        break
                    else:
                        r -= self.A_start[j]
                states.append(j)
                
            else:
                prev = states[-1]
                r = random.random()
                for j in range(self.L):
                    if r < self.A[prev][j]:
                        break
                    else:
                        r -= self.A[prev][j]
                states.append(j)
                
            r = random.random()
            for x in range(self.D):
                if r < self.O[j][x]:
                    break
                else:
                    r -= self.O[j][x]
            emission.append(x)
        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
