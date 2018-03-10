from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data

def print_line(line):
    print(' '.join(reversed(line)))

sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)
hmm = unsupervised_HMM(X, 10, 1)
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