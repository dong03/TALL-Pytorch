from utils import *

class MAIN_model(nn.Module):
    def __init__(self):
        super(MAIN_model2,self).__init__()
        self.common_dim = 1024
        self.batch_size = batch_size
        self.W_fc_c3d = nn.Parameter(torch.randn([4096*3,self.common_dim]))
        self.b_fc_c3d = nn.Parameter(torch.randn([self.common_dim]))
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.W_fc_st =  nn.Parameter(torch.randn([4800,self.common_dim]))
        self.b_fc_st =  nn.Parameter(torch.randn([self.common_dim]))
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels = 4*1024,
            out_channels = 1000,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(1000),
            nn.Conv2d(in_channels = 1000,
            out_channels = 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        )
        self.mask1 = (torch.ones([self.batch_size,self.batch_size]) - 2*torch.eye(self.batch_size)).to(device)
        self.mask2 = 1.0*torch.ones((self.batch_size,self.batch_size))/self.batch_size + torch.eye(self.batch_size)
        self.l2norm = nn.functional.normalize

    def forward(self,fv,fs):
        fv_ = self.dropout_layer(fv.mm(self.W_fc_c3d) + self.b_fc_c3d)
        fs_ = self.dropout_layer(fs.mm(self.W_fc_st ) + self.b_fc_st)
        fv_ = self.l2norm(fv_,p=2, dim=1, eps=1e-12, out=None)
        fs_ = self.l2norm(fs_,p=2, dim=1, eps=1e-12, out=None)
        fv_ = fv_.expand((fs.shape[0],fv.shape[0],1024))
        fs_ = fs_.expand((fv.shape[0],fs.shape[0],1024)).permute(1,0,2)
        fc = torch.cat([fv_*fs_,fv_+fs_,fv_,fs_],dim = 2)
        fc = fc.permute(2,0,1).unsqueeze(0)
        triple = self.conv_mlp(fc)
        triple = triple.squeeze(0)
        triple = triple.permute(1,2,0)
        return triple,self.mask1,self.mask2
