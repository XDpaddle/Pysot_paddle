from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
# import torch
import paddle
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder  # 解除注释
from pysot.models_car.model_builder import ModelBuilder as ModelBuilder_car
from pysot.tracker.tracker_builder import build_tracker



# torch.set_num_threads(1)  # 删

parser = argparse.ArgumentParser(description='tracking demo')
# parser.add_argument('--config', type=str, help='config file')
# parser.add_argument('--snapshot', type=str, help='model name')
# parser.add_argument('--video_name', default='', type=str,
#                     help='videos or image files')
# parser.add_argument('--config', type=str, help='config file', default='experiments\\siamrpn_r50_l234_dwxcorr\\config.yaml')
# parser.add_argument('--snapshot', type=str, help='model name', default='experiments\\siamrpn_r50_l234_dwxcorr\\trans_over.pdparams')
parser.add_argument('--config', default='experiments\\siamcar_r50\\config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='experiments\\siamcar_r50\\trans_over.pdparams', type=str,
        help='snapshot of models to eval')
# parser.add_argument('--video_name', default='demo/bag.avi', type=str,
#                     help='videos or image files')
parser.add_argument('--video_name', default='demo/output.mp4', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)  # cfg定义来自from pysot.core.config import cfg，args.config包含对应模型的设置
    cfg.CUDA = paddle.device.is_compiled_with_cuda() and cfg.CUDA
    device = paddle.device.set_device('gpu' if cfg.CUDA else 'cpu')

    # create model
    # model = ModelBuilder()
    if cfg.TRACK.TYPE == 'SiamCARTracker':
        model = ModelBuilder_car()
    else:
        model = ModelBuilder()


    # paddle.save(model.state_dict(), 'experiments\\siammask_r50_l3\\test.pdparams')  # 测试

    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)
    ckpt = paddle.load(args.snapshot)
    model.set_state_dict(ckpt)

    model.eval()
    if cfg.TRACK.TYPE == 'SiamCARTracker':
        params = getattr(cfg.HP_SEARCH ,'OTB100')
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    i = 0
    for frame in get_frames(args.video_name):
        i = i + 1
        if first_frame:
            try:
                fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
                fps = 24
                width = frame.shape[1]
                height = frame.shape[0]
                outVideo = cv2.VideoWriter('save_test_video.gif', fourcc, fps, (width, height))
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            if cfg.TRACK.TYPE == 'SiamCARTracker':
                outputs = tracker.track(frame, hp)
            else:
                outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            outVideo.write(frame)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
