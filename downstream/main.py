import numpy as np
import logging
import sys

import os
import argparse
import torch
import yaml
import random
import pandas as pd

from stats import dataset_stats
from dataloaders import create_dataloader

from models import RAK_PVM

from solver import make_scheduler
from solver import make_optimizer


from train import fit

_logger = logging.getLogger('train')


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

def load_data_to_cuda(datasetdir , dataname, device):
    csv_file = f"{dataname.lower()}.csv"
    csv_path = os.path.join(datasetdir, csv_file)
    df = pd.read_csv(csv_path)
    data = {}
    for index, row in df.iterrows():
        label = row['label']
        knowledge = row['knowledge']

        if label not in data:
            data[label] = []
        data[label].append(knowledge)
    return data


def run(cfg):
    savedir = os.path.join(cfg['RESULT']['log_dir'], cfg['DATASET']['dataname'], cfg['EXP_NAME'])
    if not os.path.exists(savedir): os.makedirs(savedir)

    datasetdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['DATASET']['datadir'], cfg['DATASET']['dataname'])
    if not os.path.exists(datasetdir): os.makedirs(datasetdir)


    log_out = savedir + '/output.log'
    sys.stdout = Logger(log_out)
    torch_seed(cfg['SEED'])

    device = ('cuda:0'
              '') if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    if cfg['MODEL']['learn_type'] == "RAK_PVM":
        model = RAK_PVM(
            modelname=cfg['MODEL']['modelname'],
            learn_type=cfg['MODEL']['learn_type'],
            num_classes=cfg['DATASET']['num_classes'],
        )

    model.to(device)
    print('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    trainset, testset = __import__('dataloaders').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
        datadir  = datasetdir,
        data_pct = cfg['DATASET']['data_pct'],
        img_size = cfg['DATASET']['img_size'],
        mean     = cfg['DATASET']['mean'],
        std      = cfg['DATASET']['std']
    )

    trainloader = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = make_optimizer([model], cfg['OPTIMIZER'])
    if cfg['TRAINING']['use_scheduler']:
        scheduler = make_scheduler(optimizer, cfg['SCHEDULER'])
    else:
        scheduler = None

    fit(model        = model,
        trainloader  = trainloader,
        testloader   = testloader,
        criterion    = criterion,
        optimizer    = optimizer,
        scheduler    = scheduler,
        epochs       = cfg['TRAINING']['epochs'],
        savedir      = savedir,
        log_interval = cfg['TRAINING']['log_interval'],
        device       = device,
        use_wandb    = cfg['TRAINING']['use_wandb'],
        learn_type   = cfg['MODEL']['learn_type']
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RAK_PVM')

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0, 1, 2, 3], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")  # 实验重复的次数。
    parser.add_argument('--overwrite', type=int, default=1, metavar='N', help='Train regardless of whether saved model exists')  # 是否覆盖已存在的模型。
    parser.add_argument('--no_wandb', action='store_false', help='no use wandb')
    # Model Args
    parser.add_argument('--modelname', type=str, default='vit_base_patch16_224', help='model name')
    parser.add_argument('--learn_type', type=str, default=None, choices=["RAK_PVM"], help='learn type')
    parser.add_argument('--prompt_dropout', type=float, default=0.0, help='prompt dropout rate')
    parser.add_argument('--dataname', type=str, default=None, choices=[ "mimic5_200_u", "mimic5_200_p", 'unseenchestx10', 'seenchestx10',
                                                                                 'unseenchestxray14', 'seenchestxray14'], help='data name')
    parser.add_argument('--data_pct', type=str, default=1, help='data select')
    # Config Arg
    parser.add_argument('--config', type=str, default="configs/rakpvm.yaml", help="yaml experiment config input")

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    d_stats = dataset_stats[args.dataname.lower()]

    cfg['MODEL']['modelname'] = args.modelname
    cfg['MODEL']['learn_type'] = args.learn_type
    cfg['MODEL']['prompt_dropout'] = args.prompt_dropout

    cfg['DATASET']['num_classes'] = d_stats['num_classes']
    cfg['DATASET']['dataname'] = args.dataname
    cfg['DATASET']['data_pct'] = args.data_pct
    cfg['DATASET']['img_size'] = d_stats['img_size']
    cfg['DATASET']['mean'] = d_stats['mean']
    cfg['DATASET']['std'] = d_stats['std']
    cfg['TRAINING']['use_wandb'] = args.no_wandb

    cfg['EXP_NAME'] = f"{args.modelname}-seed{cfg['SEED']}-{args.learn_type}-data_pct{args.data_pct}"

    run(cfg)












