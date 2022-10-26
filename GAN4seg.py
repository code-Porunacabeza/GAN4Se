# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
from torch.utils.data import DataLoader
import torch
import SimpleITK as sitk
from evaluation.evaluation import DSC, HDAVD
from models.net import DenseBiasNet,NetD
from utils.dataloader import DatasetFromFolder3D
from utils.loss import mix_loss

class AverageMeter(object):
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

def train_epoch(net_S,net_D, opt_S,opt_D, loss_S, dataloader, epoch, n_epochs, Iters):
    loss_S_log = AverageMeter()
    loss_G_log=AverageMeter()
    loss_D_log=AverageMeter()
    for i in range(Iters):
        input, target = next(dataloader.__iter__())
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        seg = net_S(input)

        seg=seg.detach()
        seg_masked=input.clone()
        input_mask=input.clone()
        seg_masked=input_mask*seg

        result=net_D(seg_masked)
        target_masked=input_mask*target

        target_D=net_D(target_masked)
        loss_D=torch.mean(torch.abs(result-target_D))
        loss_D.backward()
        opt_D.step()


        #train S
        net_S.zero_grad()
        seg=net_S(input)
        seg_masked=input_mask*seg

        result=net_D(seg_masked)
        target_masked=input_mask*target

        target_S=net_D(target_masked)

        errS = loss_S(seg, target)
        loss_G=torch.mean(torch.abs(result-target_S))
        loss_joint=loss_G+errS
        loss_joint.backward()
        opt_S.step()
        opt_S.zero_grad()
        loss_S_log.update(errS.data, target.size(0))
        loss_G_log.update(loss_G.data,target.size(0))
        loss_D_log.update(loss_D.data,target.size(0))

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (i + 1, Iters),
                         'DiceLoss_S %f' % (loss_S_log.avg)])

        print(res)

        print('Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (i + 1, Iters),
                         'G_Loss %f' % (loss_G_log.avg))
        print('Epoch: [%d/%d]' % (epoch + 1, n_epochs),
              'Iter: [%d/%d]' % (i + 1, Iters),
              'D_Loss %f' % (loss_D_log.avg))


    return


def train_net(n_epochs=200,
              batch_size=1,
              lr=1e-4, Iters=200,
              n_classes=5,
              crop_shape=(128, 128, 128),
              model_name="DenseBiasNet",
              train_dir='data/train',
              checkpoint_dir='weights',
              is_load=False,
              load_epoch=0,
              is_train=True):

    net_S = DenseBiasNet(n_channels=1, n_classes=n_classes)
    net_D = NetD()
    if is_load:
        net_S.load_state_dict(torch.load('{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, str(load_epoch))))

    if torch.cuda.is_available():
        net_S = net_S.cuda()
        net_D=net_D.cuda()

    if is_train:
        train_dataset = DatasetFromFolder3D(train_dir, shape=crop_shape, num_classes=5)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        opt_S = torch.optim.Adam(net_S.parameters(), lr=lr)
        opt_D = torch.optim.Adam(net_D.parameters(), lr=lr)

        loss_S = mix_loss()

        S_name=model_name+'_S'
        D_name=model_name+'_D'

        for epoch in range(n_epochs):
            train_epoch(net_S,net_D, opt_S,opt_D, loss_S, dataloader, epoch, n_epochs, Iters)
            if epoch % 10 == 0:
                torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, S_name, epoch))
                torch.save(net_D.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, net_D, epoch))
        torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, S_name, epoch))
        torch.save(net_D.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, net_D, epoch))
    return net_S

def predict(model, save_path, img_path, model_name):
    print("Predict test data")
    model.eval()
    image_filenames = [x for x in os.listdir(img_path) if is_image3d_file(x)]

    if not os.path.exists(join(save_path, model_name)):
        os.makedirs(join(save_path, model_name))
    for imagename in image_filenames:
        print(imagename)
        image = sitk.ReadImage(join(img_path, imagename))
        image = sitk.GetArrayFromImage(image)
        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image.astype(np.float32)
        image = image / 2048.
        image = image[np.newaxis, np.newaxis, :, :, :]

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            predict= model(image).data.cpu().numpy()

        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint8)
        predict = sitk.GetImageFromArray(predict)
        sitk.WriteImage(predict, join(save_path, model_name, imagename))

def is_image3d_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    n_epochs = 200
    batch_size = 1
    lr = 1e-4
    Iters = 200
    n_classes = 5
    crop_shape = (128, 128, 128)
    model_name = "GANDenseBiasNet"
    train_dir = '/root/autodl-tmp/kipa/train'
    pred_dir = 'results'
    checkpoint_dir = 'weights'
    test_dir = 'data/open'

    net_S = train_net(n_epochs=n_epochs,
                      batch_size=batch_size,
                      lr=lr,
                      Iters=Iters,
                      n_classes=n_classes,
                      crop_shape=crop_shape,
                      model_name=model_name,
                      train_dir=train_dir,
                      checkpoint_dir=checkpoint_dir)
    # predict(net_S, pred_dir, test_dir + '/image', model_name)
    # HDAVD(model_name, n_classes, pred_dir, test_dir + '/label')
    # DSC(model_name, n_classes, pred_dir, test_dir + '/label')