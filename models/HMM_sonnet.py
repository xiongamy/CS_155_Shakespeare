from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data

def print_line(line):
    print(' '.join(reversed(line)))

sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)
hmm = unsupervised_HMM(X, 10, 10)
pairs = []
for i in range(7):
    pair = hmm.generate_emission(10, rd, sd)
    pairs.append(pair)
print(pairs)

for stanza in range(3):
    print_line(pairs[stanza * 2][0])
    print_line(pairs[stanza * 2 + 1][0])
    print_line(pairs[stanza * 2][1])
    print_line(pairs[stanza * 2 + 1][1])
print(' '.join(pairs[-1][0]))
print(' '.join(pairs[-1][1]))
    
    
'''    
lines perjured spending but truth of too
streams true in stop death when or fury
in skill subscribes the impeached remembrance
of gay i of do like appear remains
vowing black increase to she be for thing day
thy tables thee thou which wretched every
nought false world-without-end pity rest in
these debateth find and show their from
your his my (might picture to eyes but
his their my thus april's eyes my heart no
nightly sit thy mind will is be doth my
saying self ocean to end moan in it
bred discourse art sun's my prove fair ears it
new-appearing the give have seen but for
'''