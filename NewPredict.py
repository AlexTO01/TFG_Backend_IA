# Predicting the squat validity
import sys

import cv2
import os
import torch
import argparse
import numpy as np
import keras

import movenet.models.model_factory
from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet_lightning", choices=["movenet_lightning", "movenet_thunder"])
# parser.add_argument('--size', type=int, default=192)
parser.add_argument('--conf_thres', type=float, default=0.3)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

if args.model == "movenet_lightning":
    args.size = 192
    args.ft_size = 48
else:
    args.size = 256
    args.ft_size = 64


def main ():

    model = movenet.models.model_factory.load_model(args.model, ft_size=args.ft_size)
    # model = model.cuda()
    Squat_result = np.array(['valid', 'invalid'])
    label_map = {label: num for num, label in enumerate(Squat_result)}
    labels = []
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir + "prueba") if
        f.is_file() and f.path.endswith(('.png', '.jpg', 'jpeg'))]

    total_kp = []
    for f in filenames:
        input_image, draw_image = read_imgfile(
            f, args.size)

        height, width, _ = draw_image.shape
        with torch.no_grad():
            input_image = torch.Tensor(input_image) # .cuda()

            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()
            kpt = kpt_with_conf.copy()
            keypoint_s = kpt[:, 2]
            keypoint_c = kpt[:, :2]
            keypoint_c[:, 0] = keypoint_c[:, 0] * height # eje Y
            keypoint_c[:, 1] = keypoint_c[:, 1] * width # eje X
            count = 0
            kp_image = []
            total_kp = []
            for ks, kc in zip(keypoint_s, keypoint_c):
                count = count + 1
                if ks < args.conf_thres:
                    continue
                #print(cv2.KeyPoint(kc[1], kc[0], 5).pt)
                kp_image.append(cv2.KeyPoint(kc[1], kc[0], 5).pt)

            if len(kp_image) == 17:
                total_kp.append(kp_image)
            else:
                print("La foto no se ha hecho correctamente, intentelo de nuevo con un angulo o calidad mejor")
                #sys.exit()

        if args.output_dir:
            draw_image = draw_skel_and_kp(
                draw_image, kpt_with_conf, conf_thres=args.conf_thres)

            cv2.imwrite(os.path.join(args.output_dir+"prueba", os.path.relpath(f, args.image_dir+"prueba")), draw_image)


    saved_model = keras.models.load_model('model.h5')
    res = saved_model.predict(total_kp)
    print(res)
    print(Squat_result[np.argmax(res)])



if __name__ == "__main__":
    main()
