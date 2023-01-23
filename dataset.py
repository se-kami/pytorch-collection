from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        # read data
        self.data = data
        # tokenize text
        # convert tokens to indices

    # must implement
    def __get_item__(self, i):
        return self.sequences[i], self.targets[i]

    # must implement
    def __len(self):
        return len(self.sequences)

dataset = MyDataset(data)
