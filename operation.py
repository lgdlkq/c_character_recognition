'''
@env_version: python3.5.2
@Author: 雷国栋
@LastEditors: 雷国栋
@Date: 2020-02-14 16:17:18
@LastEditTime: 2020-02-23 16:31:04
'''
import torch
import numpy as np
from tensorboardX import SummaryWriter
from data.dataset import Get_data
from model import net
import random
from baseset import configs
from baseset.progress_bar import ProgressBar
import os
import time
import shutil

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed_all(configs.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpus
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold = 0

if not os.path.exists(configs.weights):
    os.mkdir(configs.weights)
if not os.path.exists(configs.best_models):
    os.mkdir(configs.best_models)
if not os.path.exists(configs.logs):
    os.mkdir(configs.logs)
if not os.path.exists(configs.weights + configs.model_name + os.sep +
                      str(fold) + os.sep):
    os.makedirs(configs.weights + configs.model_name + os.sep + str(fold) +
                os.sep)
if not os.path.exists(configs.best_models + configs.model_name + os.sep +
                      str(fold) + os.sep):
    os.makedirs(configs.best_models + configs.model_name + os.sep + str(fold) +
                os.sep)

gd = Get_data("./data_split_list/trains.txt", "./data_split_list/valids.txt",
              "./data_split_list/tests.txt")


class Operater():
    def __init__(self, model, criterion, opt=None):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.opt = opt
        self.true = 0
        self.false = 0

    def train(self, resume=False):
        self.true, self.false = 0, 0
        global fold
        best_precision = 100
        start_epoch = 0
        if resume:
            checkpoint = torch.load(configs.base_side + configs.model_name +
                                    os.sep + st(fold) + '/best.pth.tar')
            start_epoch = checkpoint['epoch']
            fold = checkpoint['fold']
            best_precision = checkpoint['best_precision']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer'])
        total = len(gd.get_train_data())
        for epoch in range(start_epoch, configs.epochs):
            train_progressor = ProgressBar(mode="Train",
                                           epoch=epoch,
                                           total_epoch=configs.epochs,
                                           model_name=configs.model_name,
                                           total=total)

            epoch_loss = 0
            start = time.time()
            self.model.train()
            td = gd.get_train_data()
            for i, (x, y) in enumerate(td):
                x = x.float()
                y = y.long()
                x = x.to(device)
                y = y.to(device)
                y = y.squeeze(-1)
                train_progressor.current = i
                self.opt.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                time_cost = time.time() - start
                train_progressor.current_loss = loss.item()
                train_progressor.current_acc = self.accuracy(y, predicted)
                train_progressor()
            iter = total
            train_progressor.done(time_cost, epoch_loss / iter,
                                  self.true / (self.true + self.false) * 100.0)
            # writer.add_scalar('avg_epoch_train_loss', epoch_loss / iter, epoch)
            # writer.add_scalar('avg_epoch_train_acc', self.true / (self.true + self.false) * 100.0, epoch)
            val_loss, val_acc = self.evaluate(epoch)
            # writer.add_scalar('avg_epoch_val_loss', val_loss, epoch)
            # writer.add_scalar('avg_epoch_val_acc', val_acc, epoch)
            is_best = val_loss < best_precision
            best_precision = min(val_loss, best_precision)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_name": configs.model_name,
                    "state_dict": self.model.state_dict(),
                    "best_precision": best_precision,
                    "optimizer": self.opt.state_dict(),
                    "fold": fold,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                }, is_best, fold)

    def evaluate(self, epoch):
        val_loss = 0
        self.true, self.false = 0, 0
        total = len(get_valid_data())
        val_progressor = ProgressBar(mode="Valid",
                                     epoch=epoch,
                                     total_epoch=configs.epochs,
                                     model_name=configs.model_name,
                                     total=total)
        self.model.eval()
        vd = gd.get_valid_data()
        with torch.no_grad():
            start = time.time()
            for i, (x, y) in enumerate(vd):
                x = x.float()
                y = y.long()
                x = x.to(device)
                y = y.to(device)
                y = y.squeeze(-1)
                outputs = self.model(x)
                val_progressor.current = i
                loss = self.criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                time_cost = time.time() - start
                val_progressor.current_loss = loss.item()
                val_progressor.current_acc = self.accuracy(y, predicted)
                val_progressor()
            iter = total
            val_progressor.done(time_cost, val_loss / iter,
                                self.true / (self.true + self.false) * 100.0)
        return val_loss / iter, self.true / (self.true + self.false) * 100.0

    def test(self, td, o):
        self.true, self.false = 0, 0
        test_loss = 0
        total = len(td)
        test_progressor = ProgressBar(mode="Test",
                                      epoch=0,
                                      total_epoch=1,
                                      model_name=configs.model_name,
                                      total=total)
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            for i, (x, y) in enumerate(td):
                x = x.float()
                y = y.long()
                x = x.to(device)
                y = y.to(device)
                y = y.squeeze(-1)
                outputs = self.model(x)
                test_progressor.current = i
                _, predicted = torch.max(outputs.data, 1)
                acc = self.accuracy(y, predicted)
                loss = self.criterion(outputs, y)
                test_loss += loss.item()
                time_cost = time.time() - start
                test_progressor.current_loss = loss.item()
                test_progressor.current_acc = acc
                test_progressor()
                # writer.add_scalar(o + 'every_test_loss', loss.item(), i)
                # writer.add_scalar(o + 'every_test_acc', acc, i)
            iter = total

            test_progressor.done(time_cost, test_loss / iter,
                                 self.true / (self.true + self.false) * 100.0)

    def load_model(self):
        best_model = torch.load(configs.best_models + configs.model_name +
                                os.sep + str(fold) + '/model_best.pth.tar')
        self.model.load_state_dict(best_model["state_dict"])
        self.model.eval()

    def save_checkpoint(self, state, is_best, fold):
        filename = configs.weights + configs.model_name + os.sep + str(
            fold) + os.sep + "_checkpoint.pth.tar"
        torch.save(state, filename)
        if is_best:
            message = configs.best_models + configs.model_name + os.sep + str(
                fold) + os.sep + 'best.pth.tar'
            print("Get Better acc : %s from epoch %s saving weights to %s" %
                  (state["best_precision"], str(state["epoch"]), message))
            writepath = '../logs/%s.txt' % configs.model_name
            mode = 'a' if os.path.exists(writepath) else 'w'
            with open(writepath, mode) as f:
                print(
                    "Get Better acc : %s from epoch %s saving weights to %s" %
                    (state["best_precision"], str(state["epoch"]), message),
                    file=f)
            shutil.copyfile(filename, message)

    def accuracy(self, y, p):
        count = 0
        for i, j in zip(y, p):
            if j == i:
                count += 1
                self.true += 1
            else:
                self.false += 1
        acc = count / len(y) * 1.0 * 100
        return acc


if __name__ == "__main__":
    mod = net.resNet50(100)
    # writer = SummaryWriter(log_dir=configs.tf_logs, comment='ResNet')
    crt = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mod.parameters(),
                                 lr=configs.lr,
                                 weight_decay=configs.weight_decay)
    opt = Operater(mod, crt, optimizer)
    # opt.train()
    opt.load_model()
    opt.test(gd.get_test_data(), 'o')
    # writer.close()
