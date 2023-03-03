from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import hydra
from sklearn import metrics
import sys

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
import torch.optim as optim
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
from train_segmentation import LitUnsupervisedSegmenter
sys.path.append("/content/STEGO-master copy 2/src/deeplabv3/model")
from deeplabv3 import DeepLabV3
model =  DeepLabV3('1', project_dir="/content/STEGO-master copy 2/src/deeplabv3").cuda()

pre_model = LitUnsupervisedSegmenter.load_from_checkpoint('/content/STEGO-master copy 2/src/epoch=2-step=5424.ckpt').cuda()

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




def calculate_iou(confusion_matrix):
    """
    calculate IoU (intersecion over union) for a given confusion matrix.
    """

    ious = 0
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives

        denom = true_positives + false_positives + false_negatives

        # no entries, no iou..
        if denom == 0:
            iou = float('nan')
        else:
            iou = float(true_positives)/denom

        ious +=(iou)

    return ious/9
def eval(val_loader, model):
        softmax = nn.Softmax(dim=1)
        model.eval()
        softmax = nn.Softmax(dim=1)
        len_val = len(val_loader)
    
        val_loader_iter = iter(val_loader)
        miou = 0
        for i_ter in tqdm(val_loader):

            batch = val_loader_iter.next()

            with torch.no_grad():
                ind = batch["ind"]
                img = batch["img"]
                label = batch["label"]

            # feats,code = net(img)
            # feats = classifier(feats)
            
            feats = model(img.cuda())
            feats = F.interpolate(feats, label.shape[-2:], mode='bilinear', align_corners=True)
            feats=softmax(feats)
            feats=feats.permute((0,2,3,1)).detach().cpu().numpy()
            feats=np.argmax(feats,axis=3)
            label = label.cpu().detach().numpy()
            label = label.flatten()
            feats = feats.flatten()
            confusion_matrix = metrics.confusion_matrix(feats, label)
            iou = (calculate_iou(confusion_matrix))
            miou += iou
        model.train()
        return miou/len_val



@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)


    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(224, False, val_loader_crop),
        target_transform=get_transform(224, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )
    softmax = nn.Softmax(dim=1)
    #val_dataset = MaterializedDataset(val_dataset)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size
    kl_loss = nn.KLDivLoss()
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False)

    model_teacher = LitsupervisedSegmenter(train_dataset.n_classes, cfg,pre_model).cuda()
    abc =torch.load("/content/STEGO-master copy 2/src/best_epoch_LIPs_and_manual_label.pt")
    model_teacher.load_state_dict(abc)
    model_teacher.eval()

    

    optim_model = optim.Adam(model.parameters(), lr=1e-4)
    if cfg.submitting_to_aml:
        gpu_args = dict(gpus=1, val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)
        # gpu_args = dict(gpus=1, accelerator='ddp', val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader) // 4:
            gpu_args.pop("val_check_interval")

    crit = torch.nn.CrossEntropyLoss()
    for epoch in range(100):
        train_loader_iter = iter(train_loader)
        val_loader_iter = iter(val_loader)
        

        for i_ter in tqdm(train_loader):

            batch = train_loader_iter.next()
        
            with torch.no_grad():
                ind = batch["ind"]
                img = batch["img"]
                label = batch["label"]

            # feats,code = net(img)
            # feats = classifier(feats)
            feats = model(img.cuda())

            feats_teacher = model_teacher(img.cuda())
            feats_teacher = F.interpolate(feats_teacher, label.shape[-2:], mode='bilinear', align_corners=True)

            feats = F.log_softmax(feats, dim=1)
            feats_teacher = softmax(feats_teacher)

            loss_kl = kl_loss(feats, feats_teacher)
            loss_entropy = loss_calc(feats,label)
            loss = loss_entropy*0.1 + loss_kl*0.9
            print('Epoch {}, Loss {}'.format(epoch,loss))

            if False:
              continue
              # optim_cls.zero_grad()
              # loss.backward(retain_graph=True)
              # optim_cls.step()
            else :
              optim_model.zero_grad()
              loss.backward()
              optim_model.step()
        miou = eval(val_loader,model)
        print('MIOU = ',miou)
        torch.save(model.state_dict(),'/content/STEGO-master copy 2/checkpoints/epoch_{}.pt'.format(epoch))

if __name__ == "__main__":
    prep_args()
    my_app()
