from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data
import sys

def print_line(line):
    print(' '.join(reversed(line)))

if len(sys.argv) < 3:
    print('Enter number of states and iterations')
    sys.exit()
    
sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)
fname = 'HMM/' + sys.argv[1] + '_states.txt'
hmm = unsupervised_HMM(X, int(sys.argv[1]), int(sys.argv[2]), fname)
pairs = []
for i in range(7):
    pair = hmm.generate_emission(10, rd, sd)
    pairs.append(pair)

for stanza in range(3):
    print_line(pairs[stanza * 2][0])
    print_line(pairs[stanza * 2 + 1][0])
    print_line(pairs[stanza * 2][1])
    print_line(pairs[stanza * 2 + 1][1])
print_line(pairs[-1][0])
print_line(pairs[-1][1])