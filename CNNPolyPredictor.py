import torch
import torchvision.models.resnet as res

class CNNPolyPredictor(torch.nn.Module):
    def __init__(self, max_num_sides=14, min_number_sides=3, im_width=600):
        super(CNNPolyPredictor, self).__init__()
        self.backbone = res.resnet34(pretrained=True)
        self.im_width = im_width
        self.max_num_sides = max_num_sides
        self.min_num_sides = min_number_sides

        self.max_num_sides = max_num_sides
        self.min_num_sides = min_number_sides

        self.fc1 = torch.nn.Linear(1000, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, (max_num_sides-min_number_sides+1))

        self.fc4 = torch.nn.Linear(1000, 1024)
        self.fc5 = torch.nn.Linear(1024, 1024)
        self.fc6 = torch.nn.Linear(1024, 1)

        self.ReLU = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.side_pred_loss = torch.nn.CrossEntropyLoss()
        self.area_loss = torch.nn.L1Loss()
        self.im_width = im_width

    def forward(self, batch):
        ims = batch["poly_ims"]
        x = self.backbone(ims)

        # side prediction branch
        x1 = self.fc1(x)
        x1 = self.ReLU(x1)
        x1 = self.fc2(x1)
        x1 = self.ReLU(x1)
        side_predictions = self.fc3(x1)

        # area prediction branch
        x2 = self.fc4(x)
        x2 = self.ReLU(x2)
        x2 = self.fc5(x2)
        x2 = self.ReLU(x2)
        area_predictions = self.fc6(x2)


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


def getCNNPolyPredictor(im_width):
    model = CNNPolyPredictor(im_width=im_width).cuda()
    return model

def getOptimizerAndLRScheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    return optimizer, lr_scheduler