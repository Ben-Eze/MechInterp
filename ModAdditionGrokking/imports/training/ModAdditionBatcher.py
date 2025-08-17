import torch


class ModAdditionBatcher:
    @staticmethod    
    def mod_addition_tensor(X, p):
        return (X[..., 0] + X[..., 1]) % p
    
    def __init__(self, device, p, frac_train, n_sample_default=None):
        self.device = device

        i_shuffled = torch.randperm(p*p)
        
        X_all = torch.zeros([p*p, 2], dtype=int)
        X_all[:, 0] = i_shuffled // p
        X_all[:, 1] = i_shuffled % p
        Y_all = ModAdditionBatcher.mod_addition_tensor(X_all, p)
        
        i_test = int(frac_train * p*p)
        
        self.X_train = X_all[:i_test].to(self.device)
        self.Y_train = Y_all[:i_test].to(self.device)
        self.n_train = len(self.X_train)

        self.X_test = X_all[i_test:].to(self.device)
        self.Y_test = Y_all[i_test:].to(self.device)
        self.n_test = len(self.X_test)

        self.n_sample_default = n_sample_default
    
    def __call__(self, test_train, n_sample=None):
        if test_train == "train":
            X = self.X_train
            Y = self.Y_train
            n_X = self.n_train
        elif test_train == "test":
            X = self.X_test
            Y = self.Y_test
            n_X = self.n_test
        else:
            raise ValueError(f"`test_train` {test_train} is invalid. Can only be 'test' or 'train'")
        
        if n_sample is None:
            n_sample = self.n_sample_default
        
        assert n_sample is not None, "cannot have n_sample and n_sample_default = None"
        assert n_sample <= n_X, f"n_sample = {n_sample} is too large (> {n_X})"
        
        i_shuffled = torch.randperm(n_sample)

        return X[i_shuffled].to(self.device), \
               Y[i_shuffled].to(self.device)