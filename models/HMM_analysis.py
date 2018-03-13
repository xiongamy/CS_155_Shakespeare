from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data
import sys
import heapq

if len(sys.argv) < 2:
    print('Enter number of states')
    sys.exit()
    
sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)
fname = 'HMM/' + sys.argv[1] + '_states.txt'
hmm = unsupervised_HMM(X, int(sys.argv[1]), 0, fname)

for i, state in enumerate(hmm.O):
    print('\nstate ' + str(i + 1))
    first_10 = sorted(enumerate(state), key=lambda x: -x[1])[:10]
    print([(sd.word_from_id(word[0]), word[1]) for word in first_10])


for i, src in enumerate(hmm.A):
    print('\nsrc state ' + str(i + 1))
    srt = sorted(enumerate(src), key=lambda x: -x[1])
    print([(dst[0] + 1, dst[1]) for dst in srt])