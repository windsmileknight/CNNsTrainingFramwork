import numpy as np
import os
import torch
import collections

'''
早停法：
分析验证模型在验证集上的误差，当模型在验证集上表现不佳次数 > patience，则停止训练并保存模型
>>>>>>>>>>>>>>>>>>>>>>>>
修改 best_epoch 的输出形式
'''


class EarlyStopping:
    """
    如果模型在验证集上效果不在提升，则停止训练。
    """

    def __init__(self, model_save_path, patience=3, val_interval=1, verbose=False, delta=0):
        self.patience = patience  # 连续多少次在验证集上表现差则停止训练
        self.verbose = verbose  # 每次保存最佳参数是否输出信息
        self.counter = 0  # 记录当前表现差的次数（即连续 patience 次表现差才会早停）
        self.delta = delta  # 定义了变化多少才算 improvement
        self.best_score = None  # 当前最小的误差
        self.early_stop = False  # 是否早停
        self.model_save_path = model_save_path  # 模型保存位置
        self.val_loss_min = np.Inf  # 保存到目前为止最小 val_loss
        self.val_interval = val_interval  # 验证间隔
        self.loss_container = collections.deque(maxlen=self.val_interval)  # 保存 validating loss
        self.step = 0  # 记录最佳 epoch

    def __call__(self, val_loss, model, epoch):
        """
        EarlyStopping 的对象可调用，主要是判断当前验证集 loss 是否足够好，是否早停。
        :param val_loss: 当前验证集 loss
        :param model: 当前模型
        :return: None
        """
        self.loss_container.append(val_loss)
        if epoch % self.val_interval != 0:
            return
        self.step += self.val_interval
        interval_loss = np.mean(list(self.loss_container))
        if self.best_score is None:  # 初始状态
            self.best_score = interval_loss
            self.save_checkpoint(interval_loss, self.model_save_path, model)
        elif self.best_score < interval_loss - self.delta:  # 若当前 valid_loss 小于目前为止最好的 loss，则表示当前 model 有提升
            self.best_score = interval_loss  # 保存最优 loss
            self.save_checkpoint(interval_loss, self.model_save_path, model)  # 保存最优模型
            self.counter = 0  # counter 清零
        else:  # 当前 model 不够好
            self.counter += 1  # 表现不好次数 + 1
            print('EarlyStopping Counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                return self.step - self.val_interval * self.counter
        return self.step

    def save_checkpoint(self, val_loss, model_save_path, model):
        """
        若当前 epoch 的 model 在验证集上的表现优于之前最优，则保存模型于 model_save_path
        :param val_loss: 验证集损失
        :param model_save_path: 模型保存路径
        :param model: 模型
        :return: None
        """
        # 保存模型
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model.state_dict(), model_save_path)

        # 输出信息
        if self.verbose:
            print('the loss of validation set decrease from {:.6f} to {:.6f}, the best parameters are saved in {}'
                  .format(self.val_loss_min, val_loss, model_save_path))
        self.val_loss_min = val_loss
