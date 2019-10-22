from __future__ import print_function, division

import time
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from .util import AverageMeter


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.learning_rate
    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            preact = (opt.distill == 'abound')
            feat_s, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        logger.log_value('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()
