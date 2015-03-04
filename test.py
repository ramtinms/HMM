import unittest
from sequence import Sequence
from HMM_params import HMM_params
from HMM import HMM

class TestHMM(unittest.TestCase):

    def setUp(self):
        self.init_param = HMM_params("hmm_init_example.param")
        seq1 = Sequence(['killer','clown'],['N','N'])
        seq2 = Sequence(['killer','problem'],['N','N'])
        seq3 = Sequence(['crazy','problem'],['A','N'])
        seq4 = Sequence(['crazy','clown'],['A','N'])
        seq5 = Sequence(['problem','crazy','clown'],['N','A','N'])
        seq6 = Sequence(['clown','crazy','killer'],['N','A','N'])
        self.training_set = [seq1,seq2,seq3,seq4,seq5,seq6]
        self.test_seq=Sequence(['killer','crazy','clown','problem'],["X"]*4)

        self.hmm = HMM()
        #seq, DP_table = self.hmm.viterbi(self.test_seq,self.init_param)
        #self.hmm.print_DP_table(DP_table)
        

        #self.hmm = make_hmm_from_file(file(HMM_FILENAME))
        #self.obs = read_observations_from_file(file(OBS_FILENAME))

    def test_decoding(self):
        test_seq=Sequence(['killer','crazy','clown','problem'],["X"]*4)
        emission = [[0,0,0,1],[0.4,0.3,0.3,0]]
        pi = [0.25,0.75]
        trans = [[0.0,1.0],[0.5,0.5]]
        states = ('A','N')
        vocab= ['clown','killer','problem','crazy']
        params = HMM_params(None,states,pi,trans,emission, vocab)
        #seq, DP_table = self.hmm.viterbi(self.test_seq,self.init_param)
        seq, DP_table = self.hmm.viterbi(self.test_seq,params)
        self.hmm.print_DP_table(test_seq,DP_table)
        self.assertAlmostEqual(DP_table[0][1][0],  0.0 ,    5)
        self.assertAlmostEqual(DP_table[1][1][0],  0.225,    5)
        self.assertAlmostEqual(DP_table[0][2][0],  0.1125,    5)
        self.assertAlmostEqual(DP_table[1][2][0],  0.0,    5)
        self.assertAlmostEqual(DP_table[0][3][0],  0.0,    5)
        self.assertAlmostEqual(DP_table[1][3][0],  0.045,    5)
        self.assertAlmostEqual(DP_table[0][4][0],  0.0,    5)
        self.assertAlmostEqual(DP_table[1][4][0],  0.00675,    5)
"""

    def test_supervised_training(self):
        
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

    def test_unsupervised_training(self):
        
        self.assertAlmostEqual(prob,           9.1276e-19, 21)
"""

if __name__ == "__main__":
    unittest.main()
