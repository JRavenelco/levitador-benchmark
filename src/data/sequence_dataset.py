import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    """
    Dataset for sequence processing.
    Maintains temporal order for calculating derivatives.
    """
    def __init__(self, arrays, seq_len):
        """
        Parameters
        ----------
        arrays : dict
            Dictionary of data arrays (numpy or torch tensors).
            All arrays must have the same length in the first dimension.
        seq_len : int
            Length of sequences to generate.
        """
        self.arrays = arrays
        self.seq_len = seq_len
        # Get length from the first array in the dictionary
        first_key = next(iter(arrays))
        self.n = len(arrays[first_key]) - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        Returns a dictionary of sequences starting at idx.
        """
        return {k: v[idx:idx+self.seq_len] for k, v in self.arrays.items()}
