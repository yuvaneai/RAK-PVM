import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score

_logger = logging.getLogger('train')


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, dataloader, criterion, optimizer, log_interval: int, device: str, learn_type) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    precision_m = AverageMeter()
    recall_m = AverageMeter()
    f1_m = AverageMeter()
    confusion_m = AverageMeter()

    end = time.time()

    model.train()
    optimizer.zero_grad()
    for idx, (image, target) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        image, target = image.to(device), target.to(device)

        targets = target.clone().detach().view(-1)

        if learn_type == 'TCPA':
            outputs, dist = model(image, is_train=True)
            outputs = outputs.float()
            targets = targets.long()
            loss = criterion(outputs, targets)
            if dist is not None:
                loss += 0.1 * dist
        else:
            outputs = model(image, is_train=True)
            outputs = outputs.float()
            targets = targets.long()
            loss = criterion(outputs, targets)

        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        preds = outputs.argmax(dim=1)
        y_true = targets.cpu()
        y_pred = preds.cpu()
        acc_m.update(targets.eq(preds).sum().item() / targets.size(0), n=targets.size(0))
        recall_m.update(recall_score(y_true, y_pred, average='macro'))
        f1_m.update(f1_score(y_true, y_pred, average='macro'))
        batch_time_m.update(time.time() - end)

        if idx % log_interval == 0:
            print('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                  'Acc: {acc.avg:.3%} '
                  'LR: {lr:.3e} '
                  'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                idx + 1, len(dataloader),
                loss=losses_m,
                acc=acc_m,
                lr=optimizer.param_groups[0]['lr'],
                batch_time=batch_time_m,
                rate=image.size(0) / batch_time_m.val,
                rate_avg=image.size(0) / batch_time_m.avg,
                data_time=data_time_m))

        end = time.time()
    return OrderedDict([('acc', acc_m.avg), ('loss', losses_m.avg)])


def test(model, dataloader, criterion, log_interval: int, device: str, learn_type) -> dict:
    correct = 0
    total = 0
    total_loss = 0

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    model.eval()
    with torch.no_grad():
        for idx, (image, target) in enumerate(dataloader):
            image, target = image.to(device), target.to(device)
            targets = target.clone().detach().view(-1)

            if learn_type == 'TCPA':
                outputs, dist = model(image, is_train=False)
                outputs = outputs.float()
                targets = targets.long()
            else:
                outputs = model(image, is_train=False)
                outputs = outputs.float()
                targets = targets.long()

            num_classes = outputs.shape[-1]

            if num_classes == 2:
                y_prob = torch.softmax(outputs, dim=1)[:, 1]
                y_pred = torch.argmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
            else:
                y_prob = F.softmax(outputs, dim=1)
                y_pred = torch.argmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            acc = correct / total
            all_y_true.extend(targets.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
            all_y_prob.extend(y_prob.cpu().numpy())

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            if idx % log_interval == 0:
                print('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' %
                      (idx + 1, len(dataloader), total_loss / (idx + 1), 100. * correct / total, correct, total))

    acc1 = accuracy_score(all_y_true, all_y_pred)
    balanced_acc = balanced_accuracy_score(all_y_true, all_y_pred)

    if num_classes == 2:
        f1 = f1_score(all_y_true, all_y_pred, average='binary')
        auc = roc_auc_score(all_y_true, all_y_prob)
        cm = confusion_matrix(all_y_true, all_y_pred)
    else:
        f1 = f1_score(all_y_true, all_y_pred, average='macro')
        y_true_onehot = label_binarize(all_y_true, classes=range(num_classes))
        auc = roc_auc_score(y_true_onehot, all_y_prob, average='macro', multi_class='ovr')
        cm = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n✅ FINAL METRICS:\n"
          f"Acc: {acc1:.4f} | Balanced Acc: {balanced_acc:.4f} | "
          f"F1: {f1:.4f} | AUC: {auc:.4f}")

    return OrderedDict([
        ('acc', float(acc)),
        ('acc1', float(acc1)),
        ('loss', float(total_loss / len(dataloader))),
        ('f1', float(f1)),
        ('Confusion_matrix', cm),
        ('auc', float(auc)),
        ('balanced_acc', float(balanced_acc))
    ])

def fit(
        model, trainloader, testloader, criterion, optimizer, scheduler,
        epochs: int, savedir: str, log_interval: int, device: str, use_wandb: bool, learn_type
) -> None:
    best_acc = 0
    best_score = 0
    step = 0
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device, learn_type)
        eval_metrics = test(model, testloader, criterion, log_interval, device, learn_type)

        step += 1

        if scheduler:
            scheduler.step()

        eval_metrics['score'] = (
                        eval_metrics['auc'] +
                        eval_metrics['f1'] +
                        eval_metrics['balanced_acc']
                ) / 3

        if best_score < eval_metrics['score']:
            state = {
                'best_epoch': int(epoch),
                'best_score': float(eval_metrics['score']),
                'best_acc': float(eval_metrics['balanced_acc']),
                'best_f1': float(eval_metrics['f1']),
                'best_auc': float(eval_metrics['auc']),
                'confusion_matrix': eval_metrics['Confusion_matrix'].tolist()
            }
            json.dump(state, open(os.path.join(savedir, f'best_results.json'), 'w'), indent=4)
            np.save(os.path.join(savedir, 'best_confusion_matrix.npy'),
                    eval_metrics['Confusion_matrix'])

            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))

            print('Best Score {0:.3%} to {1:.3%}'.format(best_score, eval_metrics['score']))

            best_score = eval_metrics['score']

    print('Best Metrics - ACC: {0:.3f}, AUC: {1:.3f}, F1: {2:.3f}, SCORE: {3:.3f} (epoch {3})'.format(
        state['best_acc'], state['best_auc'], state['best_f1'], state['best_epoch'], state['best_score']))


