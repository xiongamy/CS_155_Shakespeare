from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, get_training_data

sd = SyllableDict()
X = get_training_data(sd)
hmm = unsupervised_HMM(X, 3, 10)
for i in range(14):
    print(hmm.generate_emission(20))