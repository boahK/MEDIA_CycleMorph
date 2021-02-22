import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, help='path to images')
        self.parser.add_argument('--labelroot', type=str, help='path to labels')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--inputSize', type=str, default='160,192,224', help='input image size')
        self.parser.add_argument('--fineSize', type=str, default='160,192,224', help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=2, help='# of input channels')
        self.parser.add_argument('--encoder_nc', type=str, default='16,32,32,32,32', help='# of each channels of encoder')
        self.parser.add_argument('--decoder_nc', type=str, default='32,32,32,8,8,3', help='# of each channels of decoder')
        self.parser.add_argument('--which_model_net', type=str, default='registUnet', help='selects model to use for netG')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='cycleMorph', help='chooses which model to use.')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--display_winsize', type=int, default=192,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--saveOpt', type=int, default=1)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        inputSize = self.opt.inputSize.split(',')
        self.opt.inputSize = []
        for size in inputSize:
            self.opt.inputSize.append(int(size))

        fineSize = self.opt.fineSize.split(',')
        self.opt.fineSize = []
        for size in fineSize:
            self.opt.fineSize.append(int(size))

        encoder_nc = self.opt.encoder_nc.split(',')
        self.opt.encoder_nc = []
        for enc_ch in encoder_nc:
            self.opt.encoder_nc.append(int(enc_ch))

        decoder_nc = self.opt.decoder_nc.split(',')
        self.opt.decoder_nc = []
        for dec_ch in decoder_nc:
            self.opt.decoder_nc.append(int(dec_ch))

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if self.opt.saveOpt:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
