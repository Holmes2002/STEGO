from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from torch.autograd import Variable
import torch.onnx

import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
from train_segmentation import LitUnsupervisedSegmenter

pre_model = LitUnsupervisedSegmenter.load_from_checkpoint('/content/STEGO-master copy 2/src/epoch=1-step=3171(2).ckpt').cuda()

def loss_calc(pred, label, gpu = 1):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    return criterion(pred, label)
class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv1 = nn.Conv2d(70,128,kernel_size = 3,stride =1, padding =1)
        self.conv2 = nn.Conv2d(128,256,kernel_size = 3,stride =1, padding =1)
        self.conv3 = nn.Conv2d(256,512,kernel_size = 3,stride =1, padding =1)
        self.conv4 = nn.Conv2d(512,512,kernel_size = 3,stride =1, padding =1)
        self.conv5 = nn.Conv2d(512,1024,kernel_size = 3,stride =1, padding =1)
        self.relu = nn.ReLU()


        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out_1 = self.relu(self.conv1(x))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.relu(self.conv3(out_2))
        out_4 = self.relu(self.conv4(out_3))
        out_5 = self.relu(self.conv5(out_4))
        out = self.conv2d_list[0](out_5)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](out_5)
            return out

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
         return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge']

class LitsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg,pre_model):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))
        self.layers = []
        self.net = pre_model.net
        self.classifier = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],9)
        import torch.optim as optim
        self.optim =  optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-6)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        feats,code  = self.net(x)
        x = self.classifier(code)
        
        return x

@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:

    model = LitsupervisedSegmenter(train_dataset.n_classes, cfg,pre_model).cuda()
    # abc =torch.load("/content/STEGO-master copy 2/src/epoch_5 (1).pt")
    # model.load_state_dict(abc)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "STEGO.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


if __name__ == "__main__":
    prep_args()
    my_app()
