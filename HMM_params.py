class HMM_params(object):
    def __init__(self, file, states=None, pi= None, transitions=None, emissions=None, vocab=None):
        """
        states - a list/tuple of states, e.g. ('start', 'hot', 'cold', 'end')
                 start state needs to be first, end state last
                 states are numbered by their order here
        transitions - the probabilities to go from one state to another
                      transitions[from_state][to_state] = prob
        emissions - the probabilities of an observation for a given state
                    emissions[state][observation] = prob
        vocab: a list/tuple of the names of observable values, in order
        """
        if file == None:
            self.states = states
            #self.real_states = states[1:-1]
            #self.start_state = 0
            #self.end_state = len(states) - 1
            self.transitions = transitions
            self.pi = pi
            self.emissions = emissions
            self.vocab = vocab 
            self.dic = {}
            self.label_dic = {} 
            counter=0
            for word in vocab:
                self.dic[word]=counter
                counter += 1
            counter = 0
            for label in states:
                self.label_dic[label]=counter
                counter+=1
        else:
            self.read_from_file(file) 

    def word_to_id(self,word):
        return self.dic[word]
    def label_to_id(self,label):
        return self.label_dic[label]
    def re_init(self,pi, transitions, emissions):
        self.pi = pi
        self.transitions = transitions
        self.emissions = emissions
        # Questions : should I add END_of_Sentence in vocabs    

    def get_states(self):
        return xrange(len(self.states))

    def get_name_of_state(self,st_n):
        return self.states[st_n]

    def get_transition_prob(self, source_state, target_state):
        return self.transitions[source_state][target_state]

    def get_start_prob(self, state):
        return self.pi[state]

    def get_emission_prob(self, state, observation):
        return self.emissions[state][self.word_to_id(observation)]

    def get_observation_name(observation):
        return vocab[self.word_to_id(observation)]

    # TODO read and write to file
    def read_from_file(self, file_name):
        return None
    #def write_to_file(self, file_name):
    
