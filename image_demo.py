import cv2
import argparse
import os

import keras.utils
import torch
import numpy as np

from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp

from sklearn.metrics import accuracy_score, confusion_matrix


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


from sklearn.model_selection import train_test_split


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


def main():

    model = load_model(args.model, ft_size=args.ft_size)
    # model = model.cuda()
    Squat_result = np.array(['valid', 'invalid'])
    label_map = {label: num for num, label in enumerate(Squat_result)}
    labels = []

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir+"valid") if f.is_file() and f.path.endswith(('.png', '.jpg', 'jpeg'))]

    valid = []
    invalid = []
    total_kp = []
    valid_kp = []
    invalid_kp = []
    for f in filenames:
        input_image, draw_image = read_imgfile(
            f, args.size)

        height, width, _ = draw_image.shape
        with torch.no_grad():
            input_image = torch.Tensor(input_image) # .cuda()

            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()
            valid.append(kpt_with_conf)
            kpt = kpt_with_conf.copy()
            keypoint_s = kpt[:, 2]
            keypoint_c = kpt[:, :2]
            keypoint_c[:, 0] = keypoint_c[:, 0] * height # eje Y
            keypoint_c[:, 1] = keypoint_c[:, 1] * width # eje X
            count = 0
            kp_image = []
            for ks, kc in zip(keypoint_s, keypoint_c):
                count = count + 1
                if ks < args.conf_thres:
                    continue
                #print(cv2.KeyPoint(kc[1], kc[0], 5).pt)
                kp_image.append(cv2.KeyPoint(kc[1], kc[0], 5).pt)

            if len(kp_image) >= 15:
                valid_kp.append(kp_image)
                total_kp.append(kp_image)
                labels.append(label_map["valid"])

        if args.output_dir:
            draw_image = draw_skel_and_kp(
                draw_image, kpt_with_conf, conf_thres=args.conf_thres)

            cv2.imwrite(os.path.join(args.output_dir+"valid", os.path.relpath(f, args.image_dir+"valid")), draw_image)

    filenames = [
        f.path for f in os.scandir(args.image_dir + "invalid") if f.is_file() and f.path.endswith(('.png', '.jpg', 'jpeg'))]

    for f in filenames:
        input_image, draw_image = read_imgfile(
            f, args.size)

        height, width, _ = draw_image.shape
        with torch.no_grad():
            input_image = torch.Tensor(input_image)  # .cuda()

            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()
            invalid.append(kpt_with_conf)
            kpt = kpt_with_conf.copy()
            keypoint_s = kpt[:, 2]
            keypoint_c = kpt[:, :2]
            keypoint_c[:, 0] = keypoint_c[:, 0] * height  # eje Y
            keypoint_c[:, 1] = keypoint_c[:, 1] * width  # eje X
            count = 0
            kp_image = []
            for ks, kc in zip(keypoint_s, keypoint_c):
                count = count + 1
                if ks < args.conf_thres:
                    continue
                # print(cv2.KeyPoint(kc[1], kc[0], 5).pt)
                kp_image.append(cv2.KeyPoint(kc[1], kc[0], 5).pt)

            if len(kp_image) >= 15:
                invalid_kp.append(kp_image)
                total_kp.append(kp_image)
                labels.append(label_map["invalid"])

        if args.output_dir:
            draw_image = draw_skel_and_kp(
                draw_image, kpt_with_conf, conf_thres=args.conf_thres)

            cv2.imwrite(os.path.join(args.output_dir + "invalid", os.path.relpath(f, args.image_dir + "invalid")),
                        draw_image)

    #print(valid)
    #print(invalid)


    # Setting callbacks for tensorboard
    log_dir = os.path.join('Logs_Retrain')
    tb_callback = TensorBoard(log_dir=log_dir)

    x = keras.preprocessing.sequence.pad_sequences(total_kp)
    y = keras.utils.to_categorical(labels).astype(int)

    print(x.shape)
    print(y.shape)

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(17, 2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(Squat_result.shape[0], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[tb_callback])

    model.save('model.h5')

if __name__ == "__main__":
    main()
