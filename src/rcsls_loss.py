import torch
    
class RCSLS(nn.Module):

    def __init__(self):
        super(RCSLS, self).__init__()
    
    def getknn(self, sc, x, y, k=10):
        sidx = sc.topk(10, 1, True)[1][:, :k] 
        f = (sc[torch.arange(sc.shape[0])[:, None], sidx]).sum()
        return f / k

    def forward(self, X_src, X_trans, Y_tgt, Z_src, Z_trans, Z_tgt , knn=10):
        f = 2 * (X_trans * Y_tgt).sum()
        fk0 = self.getknn(X_trans.mm(Z_tgt.t()), X_src, Z_tgt, knn)
        fk1 = self.getknn((Z_trans.mm(Y_tgt.t())).t(), Y_tgt, Z_src, knn)
        f = f - fk0 -fk1
        return -f / X_src.shape[0]


