from HMM import unsupervised_HMM
from HMM_utils import SyllableDict, RhymeDict, get_training_data

sd = SyllableDict()
X = get_training_data(sd)
rd = RhymeDict(X)

data = [([2728, 2572, 1407, 1374, 511, 1407, 1975, 2416, 2640, 1772, 328, 3132, 2500, 3085, 1772, 717, 2761, 2163, 3196, 1407], [6, 10, 12, 9, 11, 16, 2, 15, 18, 15, 15, 11, 18, 9, 2, 9, 10, 3, 15, 18]), ([402, 1069, 2937, 2057, 2400, 1772, 1415, 3069, 1046, 368, 394, 1454, 1374, 193, 1374, 2225, 2239, 1679, 587, 2802], [12, 13, 9, 11, 6, 2, 1, 14, 16, 2, 15, 6, 1, 8, 11, 1, 7, 8, 3, 16]), ([2730, 150, 1682, 904, 1450, 593, 1211, 3165, 768, 230, 3089, 409, 1772, 81, 3125, 2640, 2728, 1026, 3050, 2134], [5, 6, 10, 4, 16, 10, 3, 5, 5, 1, 12, 17, 2, 14, 12, 18, 8, 17, 6, 17]), ([1278, 1374, 313, 1869, 3205, 2762, 1550, 2731, 2320, 2707, 2730, 1065, 975, 1601, 1374, 1807, 14, 2802, 1813, 826], [15, 17, 16, 6, 18, 8, 11, 18, 11, 11, 17, 6, 6, 15, 5, 17, 16, 9, 13, 6]), ([1329, 2305, 2996, 402, 122, 1818, 1374, 1647, 1127, 1454, 1852, 1065, 2737, 122, 943, 1085, 387, 1988, 1852, 120], [4, 12, 13, 3, 0, 11, 17, 16, 17, 16, 18, 8, 0, 3, 11, 13, 6, 1, 2, 18]), ([2812, 1205, 1611, 150, 122, 2437, 2100, 1772, 1332, 296, 1092, 894, 3067, 2728, 627, 2802, 2746, 1763, 448, 2917], [6, 15, 17, 6, 6, 16, 12, 17, 3, 16, 16, 10, 12, 8, 3, 4, 10, 8, 11, 17]), ([1601, 2670, 1329, 2802, 232, 103, 2730, 1380, 3116, 1269, 1407, 1772, 3066, 2644, 1450, 935, 1374, 3058, 217, 2730], [5, 11, 1, 14, 9, 3, 10, 8, 17, 17, 15, 18, 13, 10, 15, 1, 13, 5, 2, 1]), ([2444, 891, 1016, 1620, 563, 2802, 101, 799, 3038, 2762, 768, 122, 3202, 3082, 1332, 2758, 2733, 14, 1065, 3066], [16, 11, 11, 9, 3, 3, 15, 1, 7, 8, 11, 18, 12, 0, 14, 16, 2, 1, 8, 17]), ([2414, 2387, 2483, 2064, 3081, 122, 831, 475, 1165, 1569, 1092, 122, 3024, 2811, 2414, 217, 3066, 2747, 122, 2812], [3, 3, 3, 8, 5, 18, 4, 7, 0, 5, 8, 19, 1, 8, 17, 0, 17, 9, 12, 0]), ([2978, 1275, 14, 2802, 1374, 1118, 1454, 1407, 3161, 2026, 951, 228, 1798, 2730, 1407, 768, 1772, 563, 939, 6], [8, 11, 7, 0, 10, 1, 6, 18, 10, 4, 10, 18, 4, 8, 5, 3, 15, 15, 1, 16]), ([1273, 775, 339, 2730, 1747, 1118, 1611, 2483, 768, 2031, 2575, 2280, 3106, 2740, 2784, 2738, 1833, 2766, 2707, 775], [2, 7, 1, 0, 15, 1, 3, 1, 9, 15, 4, 8, 5, 17, 5, 18, 10, 10, 0, 5]), ([1772, 2676, 1407, 1181, 2758, 2728, 1774, 2344, 122, 2791, 2483, 2758, 2728, 2802, 1647, 160, 2800, 12, 1374, 2491], [2, 9, 15, 15, 17, 5, 4, 17, 6, 1, 1, 14, 7, 16, 15, 17, 16, 10, 1, 7]), ([2747, 2414, 1407, 214, 1064, 2730, 217, 2765, 1808, 1167, 2523, 3069, 2802, 2500, 876, 943, 1815, 3108, 402, 1659], [0, 17, 0, 17, 5, 17, 16, 15, 4, 15, 18, 10, 5, 2, 17, 5, 18, 1, 2, 15]), ([2730, 1273, 3060, 1772, 1284, 2661, 1679, 2730, 2728, 1818, 729, 400, 1825, 14, 1454, 2728, 1619, 2970, 3067, 2132], [5, 16, 17, 2, 15, 17, 11, 13, 10, 15, 15, 18, 13, 16, 4, 7, 2, 1, 7, 16])]

for d in data:
	print(' '.join([sd.word_from_id(id) for id in d[0]]))
    
'''    
that stern in i compound in pictured show 				#summer's my bold woe sorrow whoever my devised those receives ye in
by forbidden use pretty she my in						#digest wherein flower breathes burn it i backward i rest rhetoric me cries to
the are meant evil is crow grew worst do 				#beauty why candles my age with summer's that first well rack
he i blood one you thou lend thee score tells 			#the for fear long i niggarding a to no each
him sauces wake by and not i make full it 				#of for themselves and fair forgotten buds plague of an
tongue great lose are and silent proving my 			#his bitter forth eternal where-through that days to these much check untainted
long sweet him to because alone the if 					#winters hate in my where sun is eyes i west be the
since esteemed fill love's could to almost drop 		#wealth thou do and yet who his this their a for where
should shall so prime white and earth clearer give 		#like forth and wasted tombs should be where they and tongue
virgin have a to i from it in worms 					#posterity false beauties new the in do my could fade (
hath doth borrowed the most from lose so do 			#power still ruinate win there thy then o'er thousand tells doth
my sweets in goddess this that name sees and 			#till so this that to make as title 'tis i some
they should in bath foot the be thoughts night gives 	#speed wherein to sorrow entertain fair none windows by many's
the hath what my hear survive me the that 				#not difference but now a it that love victor where-through quite
'''