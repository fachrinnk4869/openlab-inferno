"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData
import inferno
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from inferno_apps.FaceReconstruction.utils.output import save_obj, save_images, save_codes
from inferno_apps.FaceReconstruction.utils.test import test
from inferno.utils.other import get_path_to_assets
import cv2
import glob
from glob import glob
from skimage.transform import rescale, estimate_transform, warp
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_


class FAN():

    def __init__(self, device='cuda', threshold=0.5, mode='2D'):
        import face_alignment
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.flip_input = False
        if mode == '2D':
            try:
                mode = face_alignment.LandmarksType._2D
            except AttributeError:
                mode = face_alignment.LandmarksType.TWO_D
        elif mode == '2.5D':
            try:
                mode = face_alignment.LandmarksType._2halfD
            except AttributeError:
                mode = face_alignment.LandmarksType.TWO_HALF_D
        elif mode == '3D':
            try:
                mode = face_alignment.LandmarksType._3D
            except AttributeError:
                mode = face_alignment.LandmarksType.THREE_D
        else:
            raise ValueError('mode must be 2D or 3D')
        self.model = face_alignment.FaceAlignment(mode,
                                                  device=str(device),
                                                  flip_input=self.flip_input,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)

    # @profile
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out  # attempt to prevent memory leaks
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'


class TestData():
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None):
        self.max_detection = max_detection
        # if isinstance(testpath, list):
        #     self.imagepath_list = testpath
        # elif os.path.isdir(testpath):
        #     self.imagepath_list = glob(
        #         testpath + '/*.jpg') + glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        # elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
        #     self.imagepath_list = [testpath]
        # elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
        #     self.imagepath_list = video2sequence(testpath)
        # else:
        #     print(f'please check the test path: {testpath}')
        #     exit()
        # print('total {} images'.format(len(self.imagepath_list)))
        # self.imagepath_list = sorted(self.imagepath_list)
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        # add_pretrained_deca_to_path()
        # from decalib.datasets import detectors
        if face_detector == 'fan':
            self.face_detector = FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    # def __len__(self):
    #     return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center_x = right - (right - left) / 2.0
            center_y = bottom - (bottom - top) / 2.0
            # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center_x = right - (right - left) / 2.0
            center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
            # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        elif type == "mediapipe":
            old_size = (right - left + bottom - top) / 2 * 1.1
            center_x = right - (right - left) / 2.0
            center_y = bottom - (bottom - top) / 2.0
            # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        else:
            raise NotImplementedError(
                f" bbox2point not implemented for {type} ")
        if isinstance(center_x, np.ndarray):
            center = np.stack([center_x, center_y], axis=1)
        else:
            center = np.array([center_x, center_y])
        return old_size, center

    def run(self, image):
        # imagepath = str(self.imagepath_list[index])
        # imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(image)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor,
                            self.scaling_factor, 1))*255.

        h, w, _ = image.shape
        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 1:
                print('no face detected! run original image')
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type=bbox_type)
            else:
                if self.max_detection is None:
                    bbox = bbox[0]
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                    old_size, center = self.bbox2point(
                        left, right, top, bottom, type=bbox_type)
                else:
                    old_size, center = [], []
                    num_det = min(self.max_detection, len(bbox))
                    for bbi in range(num_det):
                        bb = bbox[0]
                        left = bb[0]
                        right = bb[2]
                        top = bb[1]
                        bottom = bb[3]
                        osz, c = self.bbox2point(
                            left, right, top, bottom, type=bbox_type)
                    old_size += [osz]
                    center += [c]

            if isinstance(old_size, list):
                size = []
                src_pts = []
                for i in range(len(old_size)):
                    size += [int(old_size[i] * self.scale)]
                    src_pts += [np.array(
                        [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2], [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                         [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
            else:
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                     [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        image = image / 255.
        if not isinstance(src_pts, list):
            DST_PTS = np.array(
                [[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            dst_image = warp(image, tform.inverse, output_shape=(
                self.resolution_inp, self.resolution_inp), order=3)
            dst_image = dst_image.transpose(2, 0, 1)
            return {'image': torch.tensor(dst_image).float(),
                    # 'image_name': imagename,
                    # 'image_path': imagepath,
                    # 'tform': tform,
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }
        else:
            DST_PTS = np.array(
                [[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            dst_images = []
            for i in range(len(src_pts)):
                tform = estimate_transform('similarity', src_pts[i], DST_PTS)
                dst_image = warp(image, tform.inverse, output_shape=(
                    self.resolution_inp, self.resolution_inp), order=3)
                dst_image = dst_image.transpose(2, 0, 1)
                dst_images += [dst_image]
            dst_images = np.stack(dst_images, axis=0)

            # imagenames = [imagename +
            #               f"{j:02d}" for j in range(dst_images.shape[0])]
            # imagepaths = [imagepath] * dst_images.shape[0]
            return {'image': torch.tensor(dst_images).float(),
                    # 'image_name': imagenames,
                    # 'image_path': imagepaths,
                    # 'tform': tform,
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }


def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg
    parser.add_argument('--input_folder', type=str, default=str(Path(get_path_to_assets()
                                                                     ) / "data/EMOCA_test_example_data/images/affectnet_test_examples"))
    parser.add_argument('--output_folder', type=str, default="image_output",
                        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str,
                        default='EMICA-CVT_flame2020_notexture', help='Name of the model to use.')
    # parser.add_argument('--model_name', type=str, default='EMICA_flame2020_notexture', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=str(
        Path(get_path_to_assets()) / "FaceReconstruction/models"))
    parser.add_argument('--save_images', type=bool, default=True,
                        help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False,
                        help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False,
                        help="If true, output meshes will be saved")

    args = parser.parse_args()

    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_name = args.model_name

    # 1) Load the model
    face_rec_model, conf = load_model(path_to_models, model_name)
    face_rec_model.cuda()
    face_rec_model.eval()

    cap = cv2.VideoCapture(0)
    # 2) Create a dataset
    detector = TestData(input_folder, face_detector="fan", max_detection=20)

    # 4) Run the model on the data
    # for i in auto.tqdm(range(len(dataset))):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Mirror frame secara horizontal
        frame = cv2.flip(frame, 1)
        # batch = dataset[i]
        batch = detector.run(frame)
        vals = test(face_rec_model, batch)
        visdict = face_rec_model.visualize_batch(
            batch, 0, None, in_batch_idx=None)
        # name = f"{i:02d}"
        # current_bs = batch["image"].shape[0]

        # for j in range(current_bs):
        #     name = batch["image_name"][j]

        #     sample_output_folder = Path(output_folder) / name
        #     sample_output_folder.mkdir(parents=True, exist_ok=True)

        # if args.save_mesh:
        #     save_obj(face_rec_model, str(
        #         sample_output_folder / "mesh_coarse.obj"), vals, j)
        # if args.save_codes:
        #     save_codes(Path(output_folder), name, vals, i=j)
        # vis_dict['shape_image']
        # print("Shape Image: ", visdict['shape_image'][0].shape)
        cv2.imshow('Shape Image', visdict['shape_image'][0])
        cv2.imshow('Landmarks Prediction', visdict['landmarks_pred_fan'][0])
        cv2.imshow('Landmarks Prediction fan',
                   visdict['landmarks_pred_fan'][0])
        cv2.imshow('Landmarks Prediction mediapipe',
                   visdict['landmarks_pred_mediapipe'][0])
        cv2.imshow('rendered Images',
                   visdict['predicted_image'][0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if args.save_images:
        # save_images(output_folder, name, visdict,
        #             with_detection=True, i=j)

    print("Done")


if __name__ == '__main__':
    main()
