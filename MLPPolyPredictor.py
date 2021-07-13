from collections import OrderedDict
import math
import torch

def createLayersDict(num_layers, powers, base, im_width=600):
    layers_dict = OrderedDict()
    layers_dict[0] = (im_width**2, int(math.pow(base, powers[0])))
    for i in range(1, num_layers-1):
        layers_dict[i] = (layers_dict[i-1][1], int(math.pow(base, powers[i])))
    layers_dict[i+1] = (layers_dict[i][1], 1024)
    return layers_dict

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))

class MLPPolyPredictor(torch.nn.Module):
    def __init__(self, layers_dict, max_num_sides=14, min_number_sides=3, im_width=600):
        super(MLPPolyPredictor, self).__init__()
        self.lin_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for key in layers_dict.keys():
            in_out_tuple = layers_dict[key]
            self.lin_layers.append(torch.nn.Linear(in_features=in_out_tuple[0], out_features=in_out_tuple[1], bias=False))
            self.batch_norms.append(torch.nn.BatchNorm1d(num_features=in_out_tuple[1], track_running_stats=True))
            self.ReLU = torch.nn.ReLU()
            self.im_width = im_width

        self.fc1 = torch.nn.Linear(in_features=1024, out_features=(max_num_sides-min_number_sides+1), bias=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=(max_num_sides-min_number_sides+1), track_running_stats=True)

        self.fc2 = torch.nn.Linear(in_features=1024, out_features=1, bias=False)
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=1, track_running_stats=True)

        self.side_pred_loss = torch.nn.CrossEntropyLoss()
        self.area_loss = torch.nn.L1Loss()
        self.max_num_sides = max_num_sides
        self.min_num_sides = min_number_sides

    def forward(self, batch):
        x = batch["poly_ims"][:, 0, :, :]
        x = torch.where(x != 255, torch.tensor(0.0, dtype=torch.float).cuda(), x)
        x = torch.nn.Flatten(start_dim=1)(x)

        for lin_layer, batch_norm in zip(self.lin_layers, self.batch_norms):
            x = lin_layer(x)
            x = batch_norm(x)
            x = self.ReLU(x)

        # sides prediction branch
        x1 = self.fc1(x)
        side_predictions = self.batch_norm1(x1)

        # area predictions branch
        x2 = self.fc2(x)
        area_predictions = self.batch_norm2(x2)

        if self.training:
            loss_dict = {}
            loss_dict["side_pred_loss"] = self.side_pred_loss(side_predictions, batch["num_sides"])
            batch["areas"] = batch["areas"] / (self.im_width*self.im_width)
            loss_dict["area_loss"] = self.area_loss(area_predictions, batch["areas"])
            return loss_dict, None
        else:
            pred_dict = {}
            side_predictions = torch.argmax(side_predictions, dim=1)
            pred_dict["side_predictions"] = side_predictions + self.min_num_sides
            pred_dict["areas"] = torch.clamp(area_predictions, min=0.0, max=1.0)
            pred_dict["areas"] *= (self.im_width*self.im_width) # rescale
            return None, pred_dict

def createPolyMLP(im_width):
    layers_dict = createLayersDict(num_layers=5, powers=[10, 10, 10, 10], base=2, im_width=im_width)
    mlp_poly_predictor = MLPPolyPredictor(layers_dict=layers_dict, max_num_sides=14, min_number_sides=3, im_width=im_width).cuda()
    mlp_poly_predictor.apply(weights_init)
    return mlp_poly_predictor

def getOptimizer(model):
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer, None # no lr_scheduler for MLP