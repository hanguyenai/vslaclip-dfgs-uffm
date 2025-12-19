from __future__ import absolute_import
from abc import ABC
import collections
import numpy as np

import torch
from torch import nn, autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda import amp


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, input, target):
        """
        :param input: feature matrix with shape (batch_size, feat_dim)
        :param target:  ground truth labels with shape (batch_size)
        :return:
        """
        n = input.size(0)
        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'cosine':
            input = F.normalize(input, dim=-1)
            dist = - torch.matmul(input, input.t())
        else:
            raise NotImplementedError

        # For each anchor, find the hardest positive and negative
        mask = target.expand(n, n).eq(target.expand(n, n).t()).float()
        dist_ap, _ = torch.topk(dist*mask - (1-mask), dim=-1, k=1)
        dist_an, _ = torch.topk(dist*(1-mask) + mask, dim=-1, k=1, largest=False)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class SoftmaxTripletLoss(TripletLoss):
    def __init__(self, margin=0.3, distance='cosine'):
        super().__init__(margin=margin, distance=distance)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        n = input.size(0)

        if self.distance == 'cosine':
            input = F.normalize(input, dim=-1)
            dist = -torch.matmul(input, input.t())  # smaller = more similar
        else:
            raise NotImplementedError

        # pos mask, remove diagonal
        mask = target.view(n, 1).eq(target.view(1, n))  # bool
        mask.fill_diagonal_(False)
        mask = mask.float()

        # hard pos/neg
        dist_ap, _ = torch.topk(dist * mask - (1 - mask) * 1e12, dim=-1, k=1)  # (n,1)
        dist_an, _ = torch.topk(dist * (1 - mask) + mask * 1e12, dim=-1, k=1, largest=False)  # (n,1)

        # logits as similarity (since dist = -cos sim)
        logits = torch.cat([-dist_ap, -dist_an], dim=1)  # (n,2)
        logp = self.logsoftmax(logits)

        loss = (-self.margin * logp[:, 0] - (1.0 - self.margin) * logp[:, 1]).mean()
        return loss


class InfoNce(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self,
                 temperature=0.07,
                 num_instance=4):

        super(InfoNce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance

    def forward(self, features):
        """
        :param features: (B, C, T)
        :param labels: (B)
        :return:
        """
        b, c, t = features.shape
        if t == 8:
            features = features.reshape(b, c, 2, 4).transpose(1, 2).reshape(b*2, c, 4)
            b, c, t = features.shape

        ni = self.ni
        features = features.reshape(b//ni, ni, c, t).permute(0, 3, 1, 2).reshape(b//ni, t*ni, c)
        features = F.normalize(features, dim=-1)
        labels = torch.arange(0, t).reshape(t, 1).repeat(1, ni).reshape(t*ni, 1)
        # (t*ni, t*ni)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (t*ni, t*ni)
        mask_pos = (1 - torch.eye(t*ni)).cuda()
        mask_pos = (mask * mask_pos).unsqueeze(0)

        # (b//ni, t*ni, t*ni)
        cos = torch.matmul(features, features.transpose(-1, -2))

        logits = torch.div(cos, self.temperature)
        exp_neg_logits = (logits.exp() * (1-mask)).sum(dim=-1, keepdim=True)

        log_prob = logits - torch.log(exp_neg_logits + logits.exp())
        loss = (log_prob * mask_pos).sum() / (mask_pos.sum())
        loss = - loss
        return loss

class CM(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None




class CM_Hard(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


class CM_Mix_mean_hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
            
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, indexes.tolist()):
            batch_centers[index].append(instance_feature)  # 找到这个ID对应的所有实例特征

        ##### Mean
        # for index, features in batch_centers.items():
        #     feats = torch.stack(features, dim=0)
        #     features_mean = feats.mean(0)
        #     ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features_mean
        #     ctx.features[index] /= ctx.features[index].norm()
        ##### Hard
        for index, features in batch_centers.items():
            distances = []
            for feature in features:  # 计算每个实例与质心之间的距离
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            mean = torch.stack(features, dim=0).mean(0)  # 均值
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()
            
            hard = np.argmin(np.array(distances))  #  余弦距离最近的，最不相似的  
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

            # rand = random.choice(features)  # 随机选一个
        #### rand
#         for index, features in batch_centers.items():

#             features_mean = random.choice(features)

#             ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features_mean
#             ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None

def cm_mix(inputs, indexes, features, momentum=0.5):
    return CM_Mix_mean_hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemoryAMP(nn.Module, ABC):
    def __init__(self, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemoryAMP, self).__init__()
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.features = None

    def forward(self, inputs, targets, cams=None, epoch=None):
        inputs = F.normalize(inputs, dim=1).cuda()
        # if self.use_hard:
        #     outputs = cm_hard(inputs, targets, self.features, self.momentum)
        # else:
        #     outputs = cm(inputs, targets, self.features, self.momentum)

        outputs = cm_mix(inputs, targets, self.features, self.momentum)
        outputs /= self.temp
        
        mean, hard = torch.chunk(outputs, 2, dim=1)
        loss = 0.5 * (F.cross_entropy(hard, targets) + F.cross_entropy(mean, targets))
        return loss

if __name__ == '__main__':
    loss = InfoNce()
    x = torch.rand(8, 16, 4).cuda()
    y = loss(x)
    print(y)

