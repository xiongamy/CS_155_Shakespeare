from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, get_training_data

sd = SyllableDict()
X = get_training_data(sd)
hmm = unsupervised_HMM(X, 3, 10)
for i in range(14):
    arrays = hmm.generate_emission(20)
    print(' '.join([sd.word_from_id(id) for id in arrays[0]]))
    
    
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