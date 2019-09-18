import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
import mxnet as mx
import numpy as np
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from face_detection.accuracy_evaluation import predict
from face_detection.config_farm import configuration_10_320_20L_5scales_v2 as cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    
    args = parser.parse_args()
    
    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        # learner.load_state(conf, 'mobilefacenet.pth', True, True)
        learner.load_state(conf, 'ir_se50.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
        
    cap = cv2.VideoCapture(str(conf.facebank_path/args.file_name))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(str(conf.facebank_path/'{}.avi'.format(args.save_name)),
                                   cv2.VideoWriter_fourcc(*'XVID'), int(fps), (1280,720))
    
    if args.duration != 0:
        i = 0
    
    symbol_file_path = 'face_detection/symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
    model_file_path = 'face_detection/saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
    # self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
    # print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
    ctx = mx.gpu(0)
    face_detector = predict.Predict(mxnet=mx,
                             symbol_file_path=symbol_file_path,
                             model_file_path=model_file_path,
                             ctx=ctx,
                             receptive_field_list=cfg.param_receptive_field_list,
                             receptive_field_stride=cfg.param_receptive_field_stride,
                             bbox_small_list=cfg.param_bbox_small_list,
                             bbox_large_list=cfg.param_bbox_large_list,
                             receptive_field_center_start=cfg.param_receptive_field_center_start,
                             num_output_scales=cfg.param_num_output_scales)
    # print(self.model)

    while cap.isOpened():
        print('READING----------------------------')
        isSuccess,frame = cap.read()
        if isSuccess:            
            image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            # image = Image.fromarray(frame)
            try:
                # bboxes, faces = mtcnn.align_multi(image, conf.face_limit, 16)

                # if cv2.waitKey(0) == ord('q'):
                #     break
                faces, infer_time = face_detector.predict(frame, resize_scale=0.2, score_threshold=0.6, top_k=10000, \
                                                NMS_threshold=0.2, NMS_flag=True, skip_scale_branch_list=[])
                print(len(faces))
                bboxes = faces
            except Exception as e:
                print(e)
                bboxes = []
                faces = []
            if len(bboxes) == 0:
                print('no face')
                continue
            else:
                # bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                # bboxes = bboxes.astype(int)
                # bboxes = bboxes + [-1,-1,1,1] # personal choice
                img_size = 112
                margin = 20
                # faces = np.empty((len(bboxes), img_size, img_size, 3))
                faces = []
                img_h, img_w, _ = np.shape(image)
                for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2= bbox[0], bbox[1], bbox[2] ,bbox[3]
                        xw1 = max(int(x1 - margin ), 0)
                        yw1 = max(int(y1 - margin ), 0)
                        xw2 = min(int(x2 + margin ), img_w - 1)
                        yw2 = min(int(y2 + margin ), img_h - 1)
                        face =  cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
                        faces.append(Image.fromarray(face[...,::-1]))
                results, score = learner.infer(conf, faces, targets, True)
                for idx,bbox in enumerate(bboxes):
                    x1, y1, x2, y2= bbox[0], bbox[1], bbox[2] ,bbox[3]
                    xw1 = max(int(x1 - margin ), 0)
                    yw1 = max(int(y1 - margin ), 0)
                    xw2 = min(int(x2 + margin ), img_w - 1)
                    yw2 = min(int(y2 + margin ), img_h - 1)
                    bbox = [xw1, yw1, xw2,yw2]
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                frame = cv2.resize(frame, dsize=None ,fx=0.25, fy=0.25)
                cv2.imshow('window', frame)
                if cv2.waitKey(0) == ord('q'):
                    break
            video_writer.write(frame)
        else:
            break
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break        
    cap.release()
    video_writer.release()
    
