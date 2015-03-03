

class Observation:

    def __init__(self,data,label='UNDEF',features_list=[]):
        self.data = data
        self.label=label
        self.features = features_list

    def get_value(self):
        return self.data
   
    def get_label(self):
        return self.label

    def set_value(self, value):
        self.data=value

    def set_label(self, label):
        self.label = label

    def get_featuers(self):
        return self.features
