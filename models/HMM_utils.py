import re

class syllable_dict:

    def __init__(self, fname='../data/Syllable_dictionary.txt'):
        self.dict = {',' : {0}, '\'' : {0}, ':' : {0}, '.' : {0}, '?' : {0}, ';' : {0}, '(' : {0}, ')' : {0}, '!' : {0}}
        self.words = [',', '\'', ':', '.', '?', ';', '(', ')', '!']
        self.index_of = {',' : 0, '\'' : 1, ':' : 2, '.' : 3, '?' : 4, ';' : 5, '(' : 6, ')' : 7, '!' : 8}
        current_index = 9
        
        self.words_with_syllable = {0 : set(self.words)}
        self.words_with_end_syllable = {0 : set(self.words)}
        
        f = open(fname, 'r')
        for line in f:
            entries = line.split(' ')
            num_entries = len(entries)
            
            word = entries[0]
            self.words.append(word)
            
            self.index_of[word] = current_index
            current_index += 1
            
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
    
    def ending_words_with_syllable(self, num_syllables):
        return self.words_with_end_syllable[num_syllables]
    
    def words_with_syllable(self, num_syllables):
        return self.words_with_syllable[num_syllables]
    
    def syllables_of_word(self, word):
        return self.dict[word]
    
    def word_from_id(self, id):
        return self.words[id]
    
    def id_from_word(self, word):
        return self.index_of[word]
        

def get_training_data():
    '''
    Reads in the sonnets from shakespeare.txt and returns a list of lists,
    where each line in a sonnet is represented as a list whose elements
    are the words (and punctuation) in the line.
    '''
    # read lines from the Shakespearan sonnet file
    lines = []
    with open('../data/shakespeare.txt', 'r') as f:
        for line in f:
            # ignore blank lines and the sonnet's number
            if (not ' ' * 5 in line) or (line != '\n'):
                lines.append(line)
            
    # convert each line to a list of words
    data = []
    for line in lines:
        words = line.split()
        words_and_punct = []
        
        # find punctuation
        for word in words:
            parts = re.split('(\W+)', word)
            if len(parts) > 1:
                for p in parts:
                    #if p == "'":
                        # check in dictionary for word
                        
                    if p != '':
                        # add to list
                        words_and_punct.append(p)
            else:
                words_and_punct.append(word)
                
        data.append(words_and_punct)
        
    return data