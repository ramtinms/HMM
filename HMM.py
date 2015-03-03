
from sequence import Sequence
from HMM_params import HMM_params
# seting up params 

# Reading params from file
# Reading sequence from file
#Sequence()


# Learning from data (supervised, learning)
#train_set=Sequence("123456","ABCDEF")

# always start from 1

#for i in xrange(1,len(train_set)):
#    print train_set[i],train_set.get_label(i)


#def fw_bk_algorithm(self, sequence):


#def train_supervised(self, sequence):

def matrix_to_string(matrix, header=None):
    """
    Return a pretty, aligned string representation of a nxm matrix.

    This representation can be used to print any tabular data, such as
    database results. It works by scanning the lengths of each element
    in each column, and determining the format string dynamically.

    @param matrix: Matrix representation (list with n rows of m elements).
    @param header: Optional tuple or list with header elements to be displayed.
    """
    if type(header) is list:
        header = tuple(header)
    lengths = []
    if header:
        for column in header:
            lengths.append(len(column))
    for row in matrix:
        for column in row:
            i = row.index(column)
            column = str(column)
            cl = len(column)
            try:
                ml = lengths[i]
                if cl > ml:
                    lengths[i] = cl
            except IndexError:
                lengths.append(cl)

    lengths = tuple(lengths)
    format_string = ""
    for length in lengths:
        format_string += "%-" + str(length) + "s "
    format_string += "\n"

    matrix_str = ""
    if header:
        matrix_str += format_string % header
    for row in matrix:
        matrix_str += format_string % tuple(row)

    return matrix_str



class HMM:
    # helper functions 
    def _follow_backpointers(self, param, DP_table, start):
        # don't bother branching
        pointer = start[0]
        seq = [pointer]
        for t in reversed(xrange(1, len(DP_table[1]))):
            val, backs = DP_table[pointer][t]
            pointer = backs[0]
            seq.insert(0, pointer)
        return seq 


    def viterbi(self, sequence, param):
        """
        Returns the most likely sequence of labels, for a given
        sequence.
        """
        # init 
        DP_table = [ [None for j in range(len(sequence)+1)]
                          for i in range(len(param.states)) ]
        print param.get_states()
        for state in param.get_states():
            DP_table[state][0]= param.get_name_of_state(state)
            DP_table[state][1]= (param.get_start_prob(state) * param.get_emission_prob(state,sequence[1]),'S')

        # filling DP_table   
        for t in range(2, len(sequence)+1):
            for state in param.get_states():
                emission_prob = param.get_emission_prob(state, sequence[t])
                last = [(old_state, DP_table[old_state][t-1][0] * \
                                    param.get_transition_prob(old_state, state) * \
                                    emission_prob) for old_state in param.get_states()]
                highest = max(last, key=lambda p: p[1])[1]
                backs = [s for s, val in last if val == highest]

                DP_table[state][t] = (highest,backs)

        
        last = [(old_state, DP_table[old_state][-1][0]) for old_state in param.get_states()]

        # Do I need exit probability ?

        highest = max(last, key = lambda p: p[1])[1]
        backs = [s for s, val in last if val == highest]
        seq = self._follow_backpointers(param, DP_table, backs)
        self.print_DP_table(sequence, DP_table)

        return seq

    def print_DP_table(self,sequence, table):
        header = [" "]+[sequence[i] for i in xrange(1,len(sequence)+1)]
        print matrix_to_string(table,header)

    def test(self):
        test_seq=Sequence(['killer','crazy','clown','problem'],["X"]*4)
        emission = [[0,0,0,1],[0.4,0.3,0.3,0]] 
        pi = [0.25,0.75]
        trans = [[0.0,1.0],[0.5,0.5]]
        states = ('A','N')
        vocab= ['clown','killer','problem','crazy']
        params = HMM_params(None,states,pi,trans,emission, vocab)
        print self.viterbi(test_seq,params)
 
################## Supervised Training 

    def test2(self):
        seq1 = Sequence(['killer','clown'],['N','N'])
        seq2 = Sequence(['killer','problem'],['N','N'])
        seq3 = Sequence(['crazy','problem'],['A','N'])
        seq4 = Sequence(['crazy','clown'],['A','N'])
        seq5 = Sequence(['problem','crazy','clown'],['N','A','N'])
        seq6 = Sequence(['clown','crazy','killer'],['N','A','N'])
        train_set = [seq1,seq2,seq3,seq4,seq5,seq6]

        emission = [[0,0,0,1],[0.4,0.3,0.3,0]]
        pi = [0.25,0.75]
        trans = [[0.0,1.0],[0.5,0.5]]

        #init_for_unsupervised
        emission = [[0.2,0.3,0.2,0.3],[0.2,0.2,0.3,0.3]]
        pi = [0.49,0.51]
        trans = [[0.49,0.51],[0.51,0.49]]  

        states = ('A','N')
        vocab= ['clown','killer','problem','crazy']
        params = HMM_params(None,states,pi,trans,emission, vocab)
        #print self.f_i(params,train_set)
        #print self.f_i_j(params,train_set)
        #print self.f_i_o(params, train_set)
        #self.supervised_train(params,train_set)
        self.unsupervised_train(params,train_set)
        test_seq=Sequence(['killer','crazy','clown','problem'],["X"]*4)
        print self.viterbi(test_seq,params)

    def supervised_train(self, params,train_set):
        params.re_init(self.f_i(params,train_set), self.f_i_j(params,train_set), self.f_i_o(params, train_set))

    def unsupervised_train(self,params, train_set):
        """
         based on EM
        """
        # TODO iteration
        for i in xrange(3): 
            params.re_init(self.g_i(params,train_set), self.g_i_j(params,train_set), self.g_i_o(params, train_set)) 


    def get_all_possible_labels(self, seq, param):
        results = []
        possible_labels = self.get_all_permutation_with_lenght(len(seq) ,param)
        for item in possible_labels:
            results.append((item,self.get_prob_of_label(seq, item, param),))
        #print results
        return results

    def get_prob_of_label(self, seq,labels ,param):
        # TODO handle underflow
        prob = param.get_start_prob(labels[0])*param.get_emission_prob(labels[0],seq[1])
        last_state = labels[0]
        for i in xrange(1,len(labels)):
            prob *= param.get_transition_prob(last_state, labels[i])* param.get_emission_prob(labels[i], seq[i+1])
            last_state = labels[i]
        return prob

    def get_all_permutation_with_lenght(self, length ,param):
        results = []
        for c in param.get_states():
            results.append([c])
        for i in xrange(1,length):
            temp_res = []
            for c in param.get_states():
                for item in results:
                    temp_res.append(item+[c])
            results = temp_res[:]
        return results

    def g_i(self,param,train_set):
        """
        compute g(i,xl)
        """
        result = [0.0]*len(param.get_states())
        total_sum = 0
        for seq in train_set:
            label_set = self.get_all_possible_labels(seq, param)
            for (possible_label,prob) in label_set:
            #print seq[1],seq[2]
            #print seq.get_label(1)
            #print param.label_to_id(seq.get_label(1))
                result[possible_label[1]]+= prob
                total_sum += prob

        for c in param.get_states():
            result[c]= result[c]/total_sum
        print result
        return result

    def f_i(self,param,train_set):
        """
        compute f(i,xl)
        """
        result = [0.0]*len(param.get_states())
        total_sum = 0
        for seq in train_set:
            #print seq[1],seq[2]
            #print seq.get_label(1)
            #print param.label_to_id(seq.get_label(1))
            result[param.label_to_id(seq.get_label(1))]+=1
            total_sum +=1
        
        for c in param.get_states():
            result[c]= result[c]/total_sum
        return result

    def g_i_j(self,param, train_set):
        result = []
        for i in xrange(len(param.get_states())):
            temp = []
            for j in xrange(len(param.get_states())):
                temp.append(0.0)
            result.append(temp)
        total_sum = [0.0]*len(param.get_states())

        for seq in train_set:
            label_set = self.get_all_possible_labels(seq, param)
            for (possible_label,prob) in label_set:
                for j in xrange(0,len(possible_label)-1):
                #print param.label_to_id(seq.get_label(j)),param.label_to_id(seq.get_label(j+1))
                    result[possible_label[j]][possible_label[j+1]]+=prob
                    total_sum[possible_label[j]] +=prob
                #print result
        # Normalization
        #print result
        for i in param.get_states():
            for j in param.get_states():
                result[i][j] = result[i][j]/total_sum[i]
        print result
        return result

    def f_i_j(self,param, train_set):
        result = []
        for i in xrange(len(param.get_states())):
            temp = []
            for j in xrange(len(param.get_states())):
                temp.append(0.0)
            result.append(temp)
        total_sum = [0.0]*len(param.get_states())
        for seq in train_set:
            for j in xrange(1,len(seq)):
                #print param.label_to_id(seq.get_label(j)),param.label_to_id(seq.get_label(j+1))
                result[param.label_to_id(seq.get_label(j))][param.label_to_id(seq.get_label(j+1))]+=1
                total_sum[param.label_to_id(seq.get_label(j))] +=1
                #print result
        # Normalization
        #print result
        for i in param.get_states():
            for j in param.get_states():
                result[i][j] = result[i][j]/total_sum[i]
        return result

    def g_i_o(self,param, train_set):
        result = []
        for i in xrange(len(param.get_states())):
            temp = []
            for j in xrange(len(param.vocab)):
                temp.append(0.0)
            result.append(temp)
        #print result
        total_sum = [0.0]*len(param.get_states())
        for seq in train_set:
            label_set = self.get_all_possible_labels(seq, param)
            for (possible_label,prob) in label_set:

                for j in xrange(0,len(possible_label)):
                    result[possible_label[j]][param.word_to_id(seq[j+1])]+= prob
                    total_sum[possible_label[j]] += prob
        # Normalization
        for i in param.get_states():
            for j in xrange(len(param.vocab)):
                result[i][j] = result[i][j]/total_sum[i]
        print result
        return result

    def f_i_o(self,param, train_set):
        result = []
        for i in xrange(len(param.get_states())):
            temp = []
            for j in xrange(len(param.vocab)):
                temp.append(0.0)
            result.append(temp)
        #print result
        total_sum = [0.0]*len(param.get_states())
        for seq in train_set:
            for j in xrange(1,len(seq)+1):
                result[param.label_to_id(seq.get_label(j))][param.word_to_id(seq[j])]+=1
                total_sum[param.label_to_id(seq.get_label(j))] +=1
        # Normalization
        for i in param.get_states():
            for j in xrange(len(param.vocab)):
                result[i][j] = result[i][j]/total_sum[i]
        return result

    #   Trash
    #def _init_trellis(self, observed, forward=True, init_func=identity):
    #    trellis = [ [None for j in range(len(observed))]
    #                      for i in range(len(self.real_states) + 1) ]
    #
    #    if forward:
    #        v = lambda s: self.transition(0, s) * self.emission(s, observed[1])
    #    else:
    #        v = lambda s: self.transition(s, self.end_state)
    #    init_pos = 1 if forward else -1
    #    for state in self.state_nums():
    #        trellis[state][init_pos] = init_func( v(state) )

    def forward_backward():
        print "TODO"

    def smooth_counts():
        print "TODO"
 
if __name__=='__main__':
    unit = HMM()
    print "here"
    unit.test2()
