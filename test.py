import unittest
from sequence import Sequence
from HMM_params import HMM_params
from HMM import HMM

class TestHMM(unittest.TestCase):

    def setUp(self):
        self.init_param = HMM_params("test.param")
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

    def test_decoding(self):
        test_seq=Sequence(['killer','crazy','clown','problem'],["X"]*4)
        seq, DP_table = self.hmm.viterbi(self.test_seq,self.init_param)
        self.hmm.print_DP_table(test_seq,DP_table)
        self.assertAlmostEqual(DP_table[0][1][0],  0.0 ,    5)
        self.assertAlmostEqual(DP_table[1][1][0],  0.225,    5)
        self.assertAlmostEqual(DP_table[0][2][0],  0.1125,    5)
        self.assertAlmostEqual(DP_table[1][2][0],  0.0,    5)
        self.assertAlmostEqual(DP_table[0][3][0],  0.0,    5)
        self.assertAlmostEqual(DP_table[1][3][0],  0.045,    5)
        self.assertAlmostEqual(DP_table[0][4][0],  0.0,    5)
        self.assertAlmostEqual(DP_table[1][4][0],  0.00675,    5)


    def test_supervised_training(self):
        self.hmm.supervised_train(self.init_param,self.training_set)
        print "Under Construction"

    def test_unsupervised_training(self):
        number_of_iteration = 3
        self.hmm.unsupervised_train(self.init_param,self.training_set, number_of_iteration)
        print "Under Construction" 

if __name__ == "__main__":
    unittest.main()
