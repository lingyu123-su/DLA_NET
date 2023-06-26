import os
import torch

def save_model(opt, models, names, epoch, logger, others=None, best=False):
    save_root = opt.checkpoint_path
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if best:
        ckpt_path = os.path.join(save_root, 'ckpt-best.pth')
    else:
        ckpt_path = os.path.join(save_root, 'ckpt-{}.pth'.format(epoch))

    save_dict = {'epoch': epoch}
    for i in range(len(names)):
        save_dict[names[i]] = models[i].state_dict()
    if others != None:
        save_dict.update(others)
    torch.save(save_dict, ckpt_path)
    logger.info('model saved in {}'.format(ckpt_path))
