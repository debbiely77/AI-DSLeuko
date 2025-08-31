import torch
from monai.networks.nets import ResNet
class FMLayer(torch.nn.Module):
    def __init__(self, n_features, k_dim):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n_features))
        self.v = torch.nn.Parameter(torch.randn(n_features, k_dim))
    def forward(self, x):
        x=x.squeeze()
        linear = self.w * x
        batch_size = x.size(0)
        interactions = []
        for i in range(batch_size):
            feat_matrix = torch.outer(x[i], x[i])  # shape: [n_features, n_features]
            vv_matrix = torch.mm(self.v, self.v.t())  # shape: [n_features, n_features]
            interaction = feat_matrix * vv_matrix  #
            interactions.append(interaction)
        stacked_features = torch.cat([
            linear,
            torch.stack(interactions).view(batch_size, -1)  #
        ], dim=1)
        return stacked_features

class Resnet_FM(torch.nn.Module):
    def __init__(self,args=None):
        super(Resnet_FM, self).__init__()
        self.imSubModel=ResNet(spatial_dims=2, block='bottleneck', layers=[3, 4, 6, 3], block_inplanes=[64, 128, 256, 512],
               n_input_channels=5, num_classes=args.embedding_dims,feed_forward=True)
        self.digitsSubModel1 = torch.nn.Sequential(
            torch.nn.Linear(20, args.embedding_dims //2, dtype=torch.double),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(args.embedding_dims // 2, args.embedding_dims // 4, dtype=torch.double),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(args.embedding_dims // 4, args.embedding_dims//8, dtype=torch.double),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.digitsSubModel2=FMLayer(args.embedding_dims//8,args.embedding_dims//8)
        self.classifier=torch.nn.Sequential(
            torch.nn.Linear(args.embedding_dims+(args.embedding_dims//8*args.embedding_dims//8+args.embedding_dims//8),args.embedding_dims,dtype=torch.double),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(args.embedding_dims, args.num_classes,dtype=torch.double)
        )
    def forward(self, input): #input=[im,digits]
        im_feature=self.imSubModel(input[0])
        digit_feature=self.digitsSubModel1(input[1])
        digit_feature = self.digitsSubModel2(digit_feature)
        if im_feature.shape[0]==1:
            print(im_feature.shape)
            print(digit_feature.shape)
        out=self.classifier(torch.concatenate((im_feature.double(),digit_feature.squeeze(1)),dim=1))
        return out