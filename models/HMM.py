# from hw6

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
        
        for iter in range(N_iters):
            if iter % max((N_iters / 10, 1)) == 0:
                print('iteration ' + str(iter) + ' of ' + str(N_iters))
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

    def random_state_given_id(self, id):
        sum = 0
        for i in range(self.L):
            sum += self.O[i][id]
        
        r = random.uniform(0, sum)
        for i in range(self.L):
            if r < self.O[i][id]:
                break
            else:
                r -= self.O[i][id]
        return i
    
    def generate_single_line(self, total_syllables, sd, start_id):
        '''
        Generates an emissions, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            total_syllables:    number of syllables to generate.
            
            rd:                 A rhyme dictionary object
            
            sd:                 A syllable dictionary object
            
            start_id:           Id of word to start with

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        start_word = sd.word_from_id(start_id)
        emission = [start_word]
        states = [self.random_state_given_id(start_id)]
        curr_s = abs(random.sample(sd.syllables_of_word(start_word), 1)[0])
        
        while curr_s < total_syllables:
            prev = states[-1]
            valid = set()
            
            sum = 0.
            state_prob = [0.] * self.L
            for num_syl in range(total_syllables - curr_s + 1):
                for word in sd.words_with_nsyllable(num_syl):
                    id = sd.id_from_word(word)
                    if id not in valid:
                        valid.add(id)
                        for j in range(self.L):
                            p = self.A[prev][j] * self.O[j][id]
                            sum += p
                            state_prob[j] += p
                        
            r = random.uniform(0, sum)
            for j in range(self.L):
                if r < state_prob[j]:
                    break
                else:
                    r -= state_prob[j]
            states.append(j)
            
            for id in valid:
                if r < self.O[j][id]:
                    break
                else:
                    r -= self.O[j][id]
            word = sd.word_from_id(id)
            emission.append(word)
            
            
            syllables = sd.syllables_of_word(word)
            if curr_s == 0:
                curr_s += abs(random.sample(syllables, 1)[0])
            else:
                valid_s = set()
                for s in syllables:
                    if s >= 0:
                        valid_s.add(s)  
                curr_s += random.sample(valid_s, 1)[0]
        return emission
        
    def generate_emission(self, total_syllables, rd, sd):
        '''
        Generates a pair of emissions, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            total_syllables:    number of syllables to generate.
            
            rd:                 A rhyme dictionary object
            
            sd:                 A syllable dictionary object

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        
        rw = rd.get_all_rhyming_words()
        sum = 0.
        state_prob = [0.] * self.L
        for id in rw:
            for j in range(self.L):
                p = self.A_start[j] * self.O[j][id]
                sum += p
                state_prob[j] += p
        
        r = random.uniform(0, sum)
        for j in range(self.L):
            if r < state_prob[j]:
                break
            else:
                r -= state_prob[j]
        state = j
        
        for id in rw:
            if r < self.O[j][id]:
                break
            else:
                r -= self.O[j][id]
        
        first_line = self.generate_single_line(total_syllables, sd, id)
        
        
        # change later
        sample = random.sample(rd.get_rhymes(id), 2)
        if sample[0] != id:
            second_line = self.generate_single_line(total_syllables, sd, sample[0])
        else:
            second_line = self.generate_single_line(total_syllables, sd, sample[1])
        
        return first_line, second_line

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
    D = 3214

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
