import re

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