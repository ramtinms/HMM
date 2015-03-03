from observation import Observation


class Sequence:
    seq = []
    def __init__(self, list_of_data, list_of_labels):
        if len(list_of_data) > 0:
            self.seq = []
            #print list_of_data
            self.add_data(list_of_data,list_of_labels)
        #TODO error on size of list_of_labels

    def __getitem__(self, key): return self.seq[key-1].get_value()
    def __setitem__(self, key, item): self.seq[key-1].set_value(item)

    def __len__(self):
         return len(self.seq)
       
    def set_label(self, key, label): self.seq[key-1].set_label(label)
    def get_label(self, key): return self.seq[key-1].get_label()

    def add_data(self, list_of_data, list_of_labels):
        for i in xrange(len(list_of_data)):
            self.seq.append(Observation(list_of_data[i],list_of_labels[i]))
# TODO read and write to file  
