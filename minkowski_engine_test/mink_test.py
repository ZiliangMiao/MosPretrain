import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import mos4d.datasets.nusc_dataset as nusc_dataset
from mos4d.models.MinkowskiEngine.customminkunet import CustomMinkUNet

class ExampleNetwork(ME.MinkowskiNetwork):
    def m_space_n_time(self, m, n):
        return [m, m, m, n]

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=out_feat,
                kernel_size=self.m_space_n_time(3, 1),
                bias=False,
                dimension=D))

        # self.conv1 = nn.Sequential(
        #     ME.MinkowskiConvolution(
        #         in_channels=in_feat,
        #         out_channels=64,
        #         kernel_size=self.m_space_n_time(3, 1),
        #         bias=False,
        #         dimension=D),
        #     ME.MinkowskiBatchNorm(64),
        #     ME.MinkowskiReLU())

        # self.conv2 = nn.Sequential(
        #     ME.MinkowskiConvolution(
        #         in_channels=64,
        #         out_channels=128,
        #         kernel_size=self.m_space_n_time(2, 1),
        #         stride=self.m_space_n_time(2, 1),
        #         dimension=D),
        #     ME.MinkowskiBatchNorm(128),
        #     ME.MinkowskiReLU())
        # self.pooling = ME.MinkowskiGlobalPooling()
        # self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)

class MOSModel(nn.Module):
    def __init__(self, cfg: dict, n_classes: int):
        super().__init__()
        self.dt_prediction = cfg["MODEL"]["DELTA_T_PREDICTION"]
        ds = cfg["DATA"]["VOXEL_SIZE"]
        self.quantization = torch.Tensor([ds, ds, ds, self.dt_prediction])
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=n_classes, D=4)

    def forward(self, past_point_clouds):
        quantization = self.quantization.type_as(past_point_clouds[0])

        past_point_clouds = [
            torch.div(point_cloud, quantization) for point_cloud in past_point_clouds
        ]
        features = [
            0.5 * torch.ones(len(point_cloud), 1).type_as(point_cloud)
            for point_cloud in past_point_clouds
        ]
        coords, features = ME.utils.sparse_collate(past_point_clouds, features)
        tensor_field = ME.TensorField(features=features, coordinates=coords.type_as(features))

        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)

        out = predicted_sparse_tensor.slice(tensor_field)
        out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
        return out

if __name__ == "__main__":
    # load config
    cfg = yaml.safe_load(open("/home/mars/4DMOS/config/train_cfg.yaml"))
    dt = cfg["MODEL"]["DELTA_T_PREDICTION"]
    ds = cfg["DATA"]["VOXEL_SIZE"]

    # create simple network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=1, out_feat=1, D=4)  # D: dimension of coordinates
    # for param in net.parameters():
    #     print(param)

    mos_net = MOSModel(cfg, 3)
    print(mos_net)

    for name, parameters in mos_net.named_parameters():
        print(name, ": ", parameters.size())
        print(parameters)

    use_nusc_data = True
    if use_nusc_data:
        # get the nusc data loader
        data = nusc_dataset.NuscSequentialModule(cfg)
        data.setup()
        train_dataloader = data.train_dataloader()

        # loop the data loader
        for data in train_dataloader:
            # get items in batch data
            sample_data_tokens, num_curr_pts, past_point_clouds, mos_labels = data
            output = mos_net(past_point_clouds)

    else:
        # test data
        test_coords = torch.Tensor([[0, 0, 4, 0],
                                    [2, 0, 2, 0],
                                    [1, 1, 3, 0],
                                    [2, 2, 4, 0],
                                    [0, 0, 4, 1],
                                    [1, 0, 3, 1],
                                    [1, 1, 4, 1],
                                    [1, 1, 3, 1],
                                    [0, 2, 3, 1],
                                    [0, 0, 4, 2],
                                    [2, 0, 2, 2],
                                    [1, 1, 4, 2],
                                    [2, 2, 3, 2]])

        test_features = 0.5 * torch.ones(test_coords.shape[0], 1)

        coords, features = ME.utils.sparse_collate([test_coords], [test_features])  # coord: batch idx, x, y, z, t
        tensor_field = ME.TensorField(features=features, coordinates=coords.type_as(features))
        sparse_tensor = tensor_field.sparse()

        # sparse_tensor = ME.SparseTensor(feat, coords=coords)
        output = net(sparse_tensor)
