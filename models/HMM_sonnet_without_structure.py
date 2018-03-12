from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data
import sys

def print_line(line):
    print(' '.join(reversed(line)))

if len(sys.argv) < 2:
    print('Enter number of states')
    sys.exit()
    
sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)
fname = 'HMM/' + sys.argv[1] + '_states.txt'
hmm = unsupervised_HMM(X, int(sys.argv[1]), 0, fname)

for i in range(14):
    print_line([sd.word_from_id(e) for e in hmm.generate_emission_without_structure(10)])