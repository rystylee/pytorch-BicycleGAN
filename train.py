import argparse
import os
from tqdm import tqdm

import torch.utils

from dataset import AlignedDataset
from trainer import Trainer


def train(args):
    print('Loading dataset...')
    dataset = AlignedDataset('{}/{}'.format(args.data_root, args.dataset_name), args.direction, args.img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print('The number of training images: {}'.format(len(dataset)))

    trainer = Trainer(args)

    global_step = 0
    for epoch in range(1, args.n_epoch + 1):
        print('Epoch: [{}] has started!'.format(epoch))
        for i, (A, B) in tqdm(enumerate(dataloader), desc='', total=len(dataloader)):
            trainer.optimize(A, B, global_step)

            if global_step % args.save_freq == 0:
                trainer.save_weights(args.save_dir, global_step)

            global_step += 1


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_root', type=str, default='data', help='')
    parser.add_argument('--dataset_name', type=str, default='portrait', help='')
    parser.add_argument('--direction', type=str, default='AtoB', help='')
    parser.add_argument('--img_size', type=int, default=128, help='')
    parser.add_argument('--batch_size', type=int, default=2, help='')

    # Model
    parser.add_argument('--input_nc', type=int, default=3, help='')
    parser.add_argument('--output_nc', type=int, default=3, help='')
    parser.add_argument('--ndf', type=int, default=64, help='')
    parser.add_argument('--ngf', type=int, default=64, help='')
    parser.add_argument('--nef', type=int, default=64, help='')
    parser.add_argument('--nz', type=int, default=8, help='')

    # Training
    parser.add_argument('--n_epoch', type=int, default=100, help='')
    parser.add_argument('--lr', type=float, default=0.0002, help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='')
    parser.add_argument('--beta2', type=float, default=0.999, help='')
    parser.add_argument('--lambda_kl', type=float, default=0.01, help='')
    parser.add_argument('--lambda_img', type=float, default=10, help='')
    parser.add_argument('--lambda_z', type=float, default=0.5, help='')

    #
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--save_freq', type=int, default=100)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    train(args)


if __name__ == "__main__":
    main()
