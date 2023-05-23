import os
import time
import torch


class Writer:
    def __init__(self, checkpoints_dir='out/checkpoints/', log_file='out/logs/'):
        self.checkpoints_dir = checkpoints_dir
        self.log_file = log_file
        self.log_file = os.path.join(self.log_file, 'log_{:s}.txt'.format(
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Rec Loss: {:.4f}, KL Loss: {:.4f}, Control Loss: {:.4f}' \
            .format(info['current_epoch'], info['epochs'], info['t_duration'],
                    info['train_loss'], info['rec_loss'], info['kl_loss'], info['c_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler != None else 0,
            },
            os.path.join(self.checkpoints_dir, f'checkpoint1_{epoch}.pt'))
