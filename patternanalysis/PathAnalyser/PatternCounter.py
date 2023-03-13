class PatternCounter:
    def __init__(self):
        self.count = 0
        self.set = set()
        
    def __len__(self):
        return len(self.set)
    
    def add(self, pattern):
        pattern_string = ['0'] * len(pattern)

        # for i in range(len(pattern)): 
        #     if pattern[i] == 1: pattern_string[i] = '1'

        self.set.add("".join(pattern_string))
