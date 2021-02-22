import os
import numpy as np
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import torch
import scipy.io as sio
from models.networks import Dense3DSpatialTransformer
from medipy.metrics import dice
import torch.nn.functional as F

def _toTorchFloatTensor(img):
    img = torch.from_numpy(img.copy())
    return img

def _transform(dDepth, dHeight, dWidth):
    batchSize = dDepth.shape[0]
    dpt = dDepth.shape[1]
    hgt = dDepth.shape[2]
    wdt = dDepth.shape[3]

    D_mesh = torch.linspace(0.0, dpt - 1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt)
    h_t = torch.matmul(torch.linspace(0.0, hgt - 1.0, hgt).unsqueeze_(1), torch.ones((1, wdt)))
    H_mesh = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
    w_t = torch.matmul(torch.ones((hgt, 1)), torch.linspace(0.0, wdt - 1.0, wdt).unsqueeze_(1).transpose(1, 0))
    W_mesh = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)

    D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    D_upmesh = dDepth.float() + D_mesh
    H_upmesh = dHeight.float() + H_mesh
    W_upmesh = dWidth.float() + W_mesh
    return torch.stack([D_upmesh, H_upmesh, W_upmesh], dim=1)

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.nThreads = 1
    opt.batchSize = 1
    model_regist = create_model(opt)
    visualizer = Visualizer(opt)
    stn = Dense3DSpatialTransformer()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))

    datafiles = []
    dataFiles = sorted(os.listdir(opt.dataroot))
    for isub, dataName in enumerate(dataFiles):
        datafiles.append(os.path.join(opt.dataroot, dataName))

    labels = sio.loadmat(os.path.join(opt.labelroot, 'labels.mat'))['labels'][0]
    originDice = np.zeros((len(datafiles), labels.shape[0]))
    registDice = np.zeros((len(datafiles), labels.shape[0]))

    atlas = np.load(os.path.join(opt.labelroot, 'atlas_norm.npz'))
    label_vol = atlas['vol']
    label_seg = atlas['seg']

    startNum = 0
    for isub, dataFile in enumerate(datafiles[startNum:]):
        print('%d #test volume = %s' % (isub+startNum, dataFile))
        ####### Image Pre-processing ##############################################
        data = sio.loadmat(dataFile)
        data_vol = data['data_affine']
        data_seg = data['seg_affine']

        test_dataS = data_seg.transpose(2, 1, 0).astype(float)  # D W H
        nd = test_dataS.shape[0]
        nw = test_dataS.shape[1]
        nh = test_dataS.shape[2]
        test_dataS = test_dataS.reshape(1, 1, nd, nw, nh)
        batch_s = _toTorchFloatTensor(test_dataS)

        dataA = data_vol
        dataB = label_vol
        test_dataA = dataA.transpose(2, 1, 0).astype(float)
        test_dataB = dataB.transpose(2, 1, 0).astype(float)
        test_dataA = test_dataA.reshape(1, 1, nd, nw, nh)
        test_dataB = test_dataB.reshape(1, 1, nd, nw, nh)
        batch_x = _toTorchFloatTensor(test_dataA)
        batch_y = _toTorchFloatTensor(test_dataB)
        ###################################################

        test_data = {'A': batch_x, 'B': batch_y, 'path': dataFile}
        model_regist.set_input(test_data)
        model_regist.test()
        visuals = model_regist.get_test_data()
        regist_flow = visuals['flow_A'].cpu().float().numpy()[0].transpose(3, 2, 1, 0)

        global_flow = regist_flow.transpose(3, 2, 1, 0)
        global_flow = _toTorchFloatTensor(global_flow).unsqueeze(0)
        regist_data = stn(batch_x.cuda().float(), global_flow.cuda().float())
        regist_data = regist_data.cpu().float().numpy()[0, 0].transpose(2, 1, 0)

        sflow = _transform(global_flow[:, 0], global_flow[:, 1], global_flow[:, 2])
        nb, nc, nd, nw, nh = sflow.shape
        segflow = torch.FloatTensor(sflow.shape).zero_()
        segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0
        segflow[:, 1] = (sflow[:, 1] / (nw - 1) - 0.5) * 2.0
        segflow[:, 0] = (sflow[:, 2] / (nh - 1) - 0.5) * 2.0
        regist_seg = F.grid_sample(batch_s.cuda().float(), (segflow.cuda().float().permute(0, 2, 3, 4, 1)), mode='nearest')
        regist_seg = regist_seg.cpu().numpy()[0, 0].transpose(2, 1, 0)

        vals_regist, _ = dice(regist_seg, label_seg, labels=labels, nargout=2)
        vals_origin, _ = dice(data_seg, label_seg, labels=labels, nargout=2)
        registDice[isub] = vals_regist
        originDice[isub] = vals_origin
        print(np.mean(vals_regist))

        dataName = dataFile.split('\\')[-1]
        savePath = os.path.join(opt.results_dir, 'regist_' + dataName)
        result_data = {'seg_regist': regist_seg.astype('float32'),
                       'data_regist': regist_data.astype('float32'),
                       'data_field': regist_flow.astype('float32')}
        sio.savemat(savePath, result_data)


    dataName = 'OASIS_testSeg.mat'
    savePath = os.path.join(opt.results_dir, dataName)
    sio.savemat(savePath, {'Dice':registDice})

    print('registDice across all structures and subject | mean = %f, std=%f' % (np.mean(registDice), np.std(registDice)))
    print('originDice across all structures and subject | mean = %f, std=%f' % (np.mean(originDice), np.std(originDice)))

    webpage.save()



