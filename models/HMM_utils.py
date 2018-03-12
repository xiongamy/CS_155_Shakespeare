punctuation = [',', '\'', ':', '.', '?', ';', '(', ')', '!']
punctuationIDs = range(9)

class SyllableDict:
    '''
    Class representing a syllable dictionary, where each word in the dict corresponds to a
    number of syllables (as listed in the file Syllable_dictionary.txt).
    '''
    def __init__(self, fname='../data/Syllable_dictionary.txt'):
        # add punctuation marks to syllable dictionary with the number of syllables set to 0
        self.dict = {',' : {0}, '\'' : {0}, ':' : {0}, '.' : {0}, '?' : {0}, ';' : {0}, '(' : {0}, ')' : {0}, '!' : {0}}
        self.words = punctuation[:]
        self.index_of = {',' : 0, '\'' : 1, ':' : 2, '.' : 3, '?' : 4, ';' : 5, '(' : 6, ')' : 7, '!' : 8}
        current_index = 9
        
        # generate a new empty set for each possible number of syllables (1 to 10)
        self.words_with_syllable = {0 : set(self.words)}
        self.words_with_end_syllable = {0 : set(self.words)}
        for syllables in range(1, 11):
            self.words_with_syllable[syllables] = set()
            self.words_with_end_syllable[syllables] = set()
        
        f = open(fname, 'r')
        for line in f:
            # split each entry into the word and its possible number of syllables
            entries = line.split(' ')
            num_entries = len(entries)
            
            word = entries[0]
            self.words.append(word)
            
            self.index_of[word] = current_index
            current_index += 1
            
            # update syllable dictionary for the word and each of its possible
            # number of syllables
            vals = set()
            for i in range(1, num_entries):
                entry = entries[i]
                if entry[0] == 'E':
                    num_syllables = int(entry[1:])
                    vals.add(-num_syllables)
                else:
                    num_syllables = int(entry)
                    vals.add(num_syllables)
                
                    if  num_syllables not in self.words_with_syllable:
                        self.words_with_syllable[num_syllables] = set([word])
                    else:
                        self.words_with_syllable[num_syllables].add(word)
                
                if  num_syllables not in self.words_with_end_syllable:
                    self.words_with_end_syllable[num_syllables] = set([word])
                else:
                    self.words_with_end_syllable[num_syllables].add(word)
                
            
            self.dict[word] = vals
        f.close()
    
    def ending_words_with_nsyllable(self, num_syllables):
        # Returns a set with all the words that have n number of syllables
        # if it is the ending word
        return self.words_with_end_syllable[num_syllables]
    
    def words_with_nsyllable(self, num_syllables):
        # Returns a set with all the words that have n number of syllables
        return self.words_with_syllable[num_syllables]
    
    def syllables_of_word(self, word):
        # Returns a set of all possible number of syllables the given
        # word has (with negatives indicating this is only syllable
        # count if the word is at the end of the line)
        return self.dict[word]
    
    def word_from_id(self, id):
        # Returns the word (as a string) from the given ID
        return self.words[id]
    
    def id_from_word(self, word):
        # Returns the ID of the given word
        return self.index_of[word]
    
    def contains_word(self, word):
        # Returns whether the word is in the syllable dictionary
        return word in self.dict



class RhymeDict:
    '''
    Class representing a rhyming dictionary from the rhymes in the given
    training data.
    '''
    def __init__(self, training_data):
    
        def get_first_word(line):
            # skip punctuation to get the first word (returned as an ID)
            for j in range(len(line)):
                if not line[j] in punctuationIDs:
                    return line[j]
            return -1
            
        # rhyme scheme is abab cdcd efef gg
        rhyme_lines = [0, 1, 4, 5, 8, 9, 12]
        
        rhyme_sets = []
        word_to_sets = {}
        for i in range(len(training_data)):
            if not i % 14 in rhyme_lines:
                continue
                
            word = get_first_word(training_data[i])
            if i % 14 == 12:
                rhyme_word = get_first_word(training_data[i + 1])
            else:
                rhyme_word = get_first_word(training_data[i + 2])
                
            if word == -1 or rhyme_word == -1:
                continue
                
            # update the rhyme sets
            word_in_set = word in word_to_sets
            rhyme_in_set = rhyme_word in word_to_sets
            
            if word_in_set and rhyme_in_set:
                # are they in different sets?
                set_index = word_to_sets[word]
                if word_to_sets[rhyme_word] != set_index:
                    # combine sets together
                    other_index = word_to_sets[rhyme_word]
                    other_set = rhyme_sets[other_index]
                    rhyme_sets[set_index] |= other_set
                    
                    # update word_to_sets for all words in the other set
                    for key in other_set:
                        word_to_sets[key] = set_index
            elif word_in_set:
                # add rhyme_word to word's set
                set_index = word_to_sets[word]
                rhyme_sets[set_index].add(rhyme_word)
                word_to_sets[rhyme_word] = set_index
            elif rhyme_in_set:
                # add word to rhyme_word's set
                set_index = word_to_sets[rhyme_word]
                rhyme_sets[set_index].add(word)
                word_to_sets[word] = set_index
            else:
                # create new set
                new_set = set([word, rhyme_word])
                set_index = len(rhyme_sets)
                rhyme_sets.append(new_set)
                word_to_sets[word] = set_index
                word_to_sets[rhyme_word] = set_index
                
        
        self.rhyme_sets = rhyme_sets
        self.word_to_sets = word_to_sets
        

    def get_all_rhyming_words(self):
        # Get a set of all words that have rhymes.
        all_words = set()
        for set_t in self.rhyme_sets:
            all_words |= set_t
        return all_words
    
    def get_rhymes(self, word):
        # Get a set of all the words that rhyme with the given word.
        return self.rhyme_sets[self.word_to_sets[word]]
        
    def print_rhyme_sets(self, syllable_dict):
        # Print the rhyme sets as lists of strings to the console.
        rhymes_as_words = []
        for s in self.rhyme_sets:
            set_words = set([])
            for id in s:
                set_words.add(syllable_dict.word_from_id(id))
            
            rhymes_as_words.append(set_words)
            
        print(rhymes_as_words)
        

def get_training_data(syllable_dict):
    '''
    Reads in the sonnets from shakespeare.txt and returns a list of lists,
    where each line in a sonnet is represented as a list whose elements
    are the IDs of the words (and punctuation) in the line, where each
    line is in reverse order.
    '''
    
    def find_word_parts(word, l):
        '''
        Adds the words without punctuation as well as any punctuation in
        the string to the list l.
        '''
        w = word
        while len(w) > 0:
            if syllable_dict.contains_word(w):
                l.append(w)
                find_word_parts(word[len(w):], l)
                break
            else:
                w = w[:-1]
        
    # read lines from the Shakespearan sonnet file
    lines = []
    skip_sonnet = False
    with open('../data/shakespeare.txt', 'r') as f:
        for line in f:
            line = line.strip()
            
            if skip_sonnet:
                # skip lines until you reach another digit
                if line.isdigit():
                    skip_sonnet = False
                continue
            
            # skip sonnets 99 and 126
            if line == '99' or line == '126':
                skip_sonnet = True
                continue
            
            # ignore blank lines and the sonnet's number
            if not line.isdigit() and line != '':
                lines.append(line)
            
    # convert each line to a list of words
    data_strings = []
    for line in lines:
        words = line.split()
        words_and_punct = []
        
        # find punctuation
        for word in words:
            word = word.lower()
            find_word_parts(word, words_and_punct)
        
        words_and_punct.reverse()
        data_strings.append(words_and_punct)
        
    # convert the words to their IDs
    data = []
    for line in data_strings:
        l = []
        for word in line:
            l.append(syllable_dict.id_from_word(word))
        data.append(l)
        
    return data