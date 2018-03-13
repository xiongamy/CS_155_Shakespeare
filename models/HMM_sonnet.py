from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data
import sys

def print_line(line):
    print(' '.join(reversed(line)))

if len(sys.argv) < 3:
    print('Enter number of states and iterations')
    sys.exit()
    
# Pre-processing; get training data and generate syllable and rhyming dictionaries
sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)

# Load in previous HMM and train it some more
fname = 'HMM/' + sys.argv[1] + '_states.txt'
hmm = unsupervised_HMM(X, int(sys.argv[1]), int(sys.argv[2]), fname)

# Generate a sonnet as 7 pairs of rhyming lines with
# the rhyme scheme abab cdcd efef gg.
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