import argparse
import time
import os
import math
import base64
import urllib.parse
import requests
import json
import timeit
import sys
import io
import json

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

#FILE = Path(__file__).absolute()
libs_dir_path = "libs"
n_libs = 0
for filename in os.listdir(libs_dir_path):
    filepath = os.path.join(libs_dir_path, filename)
    if os.path.isdir(filepath):
        sys.path.insert(n_libs, os.path.abspath(filepath))
        n_libs += 1

from libs.yolov5.utils.torch_utils import select_device
from libs.yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from libs.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from libs.yolov5.utils.datasets import LoadStreams, LoadImages
from libs.yolov5.models.experimental import attempt_load

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from libs.vietocr.vietocr.tool.predictor import Predictor
from libs.vietocr.vietocr.tool.config import Cfg


class LoadImages1:  # for inference
    def __init__(self, img, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.img = img
        self.nf = 1
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        self.count += 1
        img0 = self.img

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return "", img, img0, self.cap

    def __len__(self):
        return self.nf  # number of files


def load_model(model_path, imgsz):
    device = ''
    device = select_device(device)
    half = False
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(
        model, 'module') else model.names  # get class names

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once

    return model, names, stride, device, half


def four_corners_detection(model, names, imgsz, stride, device, half, process_img):
    augment = False
    visualize = False
    conf_thres = 0.7
    iou_thres = 0.45
    max_det = 1000
    agnostic_nms = False
    classes = None

    dataset = LoadImages1(process_img, img_size=imgsz, stride=stride)

    return_data = []
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.cpu().tolist()
#                print(det_cpu)
                # Write results
                for *xyxy, conf, cls in reversed(det_cpu):
                    bbox = xyxy
                    conf = float(conf)
                    cls = names[int(cls)]
                    return_data.append(
                        {'bbox': bbox, 'score': conf, 'cls': cls})

    return return_data


def text_detection(model, names, imgsz, stride, device, half, process_img):
    augment = False
    visualize = False
    conf_thres = 0.7
    iou_thres = 0.45
    max_det = 1000
    agnostic_nms = False
    classes = None

    dataset = LoadImages1(process_img, img_size=imgsz, stride=stride)

    return_data = []
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.cpu().tolist()
#                print(det_cpu)
                # Write results
                for *xyxy, conf, cls in reversed(det_cpu):
                    bbox = xyxy
                    conf = float(conf)
                    cls = names[int(cls)]
                    return_data.append(
                        {'bbox': bbox, 'score': conf, 'cls': cls})

    return return_data


def resize_if_too_large(input_img):
    output_img = np.copy(input_img)
    h, w = output_img.shape[:2]
    # resize input image if its size is too large
    if w > 750:
        new_w = 750
        new_h = 750/w * h
        output_img = cv2.resize(
            output_img, (int(new_w), int(new_h)), cv2.INTER_AREA)

    return output_img


def calc_distance(point_a, point_b):
    return math.sqrt((point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2)


def get_center_point(coordinate_dict):
    di = dict()
    # corner_offset = {'top_left': [-30, -30], 'top_right': [30, -30], 'bottom_left': [-30, 30], 'bottom_right': [30, 30]}
    for obj in coordinate_dict:
        xmin, ymin, xmax, ymax = obj['bbox']
        #print(f"{obj['cls']}: {obj['bbox']}")
        x_center = int((xmin + xmax) / 2)
        y_center = int((ymin + ymax) / 2)
        # if obj['bbox'] == 'top_left':
        #     x_center = xmin
        #     y_center = ymin
        # elif obj['bbox'] == 'top_right':
        #     x_center = xmax
        #     y_center = ymin
        # elif obj['bbox'] == 'bottom_left':
        #     x_center = xmin
        #     y_center = ymax
        # else:
        #     x_center = xmax
        #     y_center = ymax
        di[obj['cls']] = [x_center, y_center,
                          int(xmax - xmin), int(ymax - ymin)]

    return di

# Crop 4 corners


def perspective_transform(image, x_ratio, y_ratio, corners):
    # Order points in clockwise order
    #ordered_corners = order_corner_points(corners)
    #top_l, top_r, bottom_r, bottom_l = ordered_corners
    top_l, top_r, bottom_r, bottom_l = corners

    h, w = image.shape[:-1]
    # offsets = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    offset = 35
    top_l[0] = max(0, top_l[0] - offset) * x_ratio
    top_l[1] = max(0, top_l[1] - offset) * y_ratio
    top_r[0] = min(w, top_r[0] + offset) * x_ratio
    top_r[1] = max(0, top_r[1] - offset) * y_ratio
    bottom_r[0] = min(w, bottom_r[0] + offset) * x_ratio
    bottom_r[1] = min(h, bottom_r[1] + offset) * y_ratio
    bottom_l[0] = max(0, bottom_l[0] - offset) * x_ratio
    bottom_l[1] = min(h, bottom_l[1] + offset) * y_ratio

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0])
                       ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) +
                      ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) +
                       ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) +
                       ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")

    # Convert to Numpy format
    ordered_corners = np.array(corners, dtype="float32")
    #print("ordered_corners", ordered_corners)
    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_4corner(resized_img, original_img, coordinate_dict):
    corner_label = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    coordinate_dict = get_center_point(coordinate_dict)

    # corners = []
    # already_appended = []
    # for predict in response:
    #     if predict['object'] in corner_label and predict['object'] not in already_appended:
    #         corners.append(predict)
    #         already_appended.append(predict['object'])
    corners = coordinate_dict.keys()
    if len(corners) <= 1:
        return None

    try:
        topleft_center = [coordinate_dict["top_left"][:2]]
    except KeyError:
        topleft_center = []
    try:
        topright_center = [coordinate_dict["top_right"][:2]]
    except KeyError:
        topright_center = []
    try:
        botright_center = [coordinate_dict["bottom_right"][:2]]
    except KeyError:
        botright_center = []
    try:
        botleft_center = [coordinate_dict["bottom_left"][:2]]
    except:
        botleft_center = []
    # print(topleft_center)
    # print(topright_center)
    # print(botright_center)
    # print(botleft_center)
    # print(len(corners))
    if len(corners) == 2:
        if len(topleft_center) > 0 and len(botright_center) > 0:
            topright_center = [[botright_center[0][0], topleft_center[0][1]]]
            botleft_center = [[topleft_center[0][0], botright_center[0][1]]]
        elif len(topright_center) > 0 and len(botleft_center) > 0:
            topleft_center = [[botleft_center[0][0], topright_center[0][1]]]
            botright_center = [[topright_center[0][0], botleft_center[0][1]]]
        else:
            return None

    if len(corners) == 3:
        if len(topleft_center) == 0:
            tl_x = int(
                topright_center[0][0] - calc_distance(botright_center[0], botleft_center[0]))
            tl_y = int(
                botleft_center[0][1] - calc_distance(botright_center[0], topright_center[0]))
            topleft_center = [[tl_x, tl_y]]
        elif len(topright_center) == 0:
            tr_x = int(
                topleft_center[0][0] + calc_distance(botright_center[0], botleft_center[0]))
            tr_y = int(
                botright_center[0][1] - calc_distance(botleft_center[0], topleft_center[0]))
            topright_center = [[tr_x, tr_y]]
        elif len(botright_center) == 0:
            br_x = int(
                botleft_center[0][0] + calc_distance(topright_center[0], topleft_center[0]))
            br_y = int(
                topright_center[0][1] + calc_distance(botleft_center[0], topleft_center[0]))
            botright_center = [[br_x, br_y]]
        elif len(botleft_center) == 0:
            bl_x = int(
                botright_center[0][0] - calc_distance(topright_center[0], topleft_center[0]))
            bl_y = int(
                topleft_center[0][1] + calc_distance(botright_center[0], topright_center[0]))
            botleft_center = [[bl_x, bl_y]]

    x_ratio = original_img.shape[1] / resized_img.shape[1]
    y_ratio = original_img.shape[0] / resized_img.shape[0]
#    print(x_ratio, y_ratio)

    corners_center = (
        topleft_center[0], topright_center[0], botright_center[0], botleft_center[0])
#    print(corners_center)
    transformed = perspective_transform(
        original_img, x_ratio, y_ratio, corners_center)
    # print(transformed.shape)
    #cv2.imshow("cropped_img", transformed)
    #cv2.imwrite('cropped_img.jpg', transformed)
    # cv2.waitKey()

    return transformed


def crop_image(coordinate_dict, resized_image, image):
    #image = cv2.imread(args.image)

    #resized_image = resize_if_too_large(image)

    cropped_image = crop_4corner(resized_image, image, coordinate_dict)

    return cropped_image

############################################################################
# Rotation


def annotation_list_to_array(coordinate_list):
    """
    Convert returned detected bounding box coordinates (from list to seperated classes, boxes arrays)
    """
    classes = []
    list_with_all_boxes = []
    for obj in coordinate_list:
        classes.append(obj['cls'])
        list_with_all_boxes.append(obj['bbox'])

    return classes, np.array(list_with_all_boxes)


def get_corners(bboxes):
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    x_left = np.min(x_, 1).reshape(-1, 1)
    y_left = np.min(y_, 1).reshape(-1, 1)
    x_right = np.max(x_, 1).reshape(-1, 1)
    y_right = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((x_left, y_left, x_right, y_right, corners[:, 8:]))

    return final


def bboxes_rotation_final(original_img, bboxes, angle):
    w, h = original_img.shape[1], original_img.shape[0]
    cx, cy = w//2, h//2

    corners = get_corners(bboxes)
    corners = np.hstack((corners, bboxes[:, 4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bboxes = get_enclosing_box(corners)
    # print(new_bboxes)
    #scale_factor_x = rotated_img.shape[1] / w
    #scale_factor_y = rotated_img.shape[0] / h
    #new_bboxes[:, :4] = np.round(new_bboxes[:, :4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])
    # print(new_bboxes)
    new_bboxes = new_bboxes.astype('int32')
    return new_bboxes.tolist()


def check_wrong_rotate_image(image, coordinate_dict):
    angle = 0
    h, w = image.shape[:2]
    h_center, w_center = h//2, w//2
    centered_dict = get_center_point(coordinate_dict)

    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for corner in corners:
        if corner in centered_dict.keys():
            continue
        centered_dict[corner] = [-1, -1, -1, -1]

    if (centered_dict["bottom_left"][0] in range(w_center + 1) and centered_dict["bottom_right"][0] in range(w_center + 1)) or (centered_dict["bottom_left"][0] in range(w_center + 1) and centered_dict["top_left"][0] in range(w_center, w + 1)) or (centered_dict["top_right"][0] in range(w_center, w + 1) and centered_dict["top_left"][0] in range(w_center, w + 1)) or (centered_dict["bottom_right"][0] in range(w_center + 1) and centered_dict["top_right"][0] in range(w_center, w + 1)):
        angle = 90
    elif (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["bottom_right"][0] in range(w_center, w + 1)) or (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["top_left"][0] in range(w_center + 1)) or (centered_dict["top_left"][0] in range(w_center + 1) and centered_dict["top_right"][0] in range(w_center + 1)) or (centered_dict["top_right"][0] in range(w_center + 1) and centered_dict["bottom_right"][0] in range(w_center, w + 1)):
        angle = -90
    elif (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["bottom_right"][0] in range(w_center + 1)) or (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["top_left"][0] in range(w_center, w + 1)) or (centered_dict["top_right"][0] in range(w_center + 1) and centered_dict["bottom_right"][0] in range(w_center + 1)) or (centered_dict["top_left"][0] in range(w_center, w + 1) and centered_dict["top_right"][0] in range(w_center + 1)):
        angle = 180

    return angle


def draw_bounding_box(image, coordinate):
    for obj in coordinate[:]:
        bbox = [int(i) for i in obj['bbox']]
        #score = obj['score']
        cls = obj['cls']
        cv2.rectangle(image, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 0, 0), 3)

        image = Image.fromarray(image)

        #ttf=ImageFont.truetype('TimesNewRoman.ttf', 30)
        ttf = ImageFont.load_default()
        ImageDraw.Draw(image).text(
            (bbox[0], bbox[1]-10), cls, fill=(0, 0, 255), font=ttf)

        image = np.asarray(image)

    return image


def check_duplicate_cls(four_corners_coordinate):
    result = []
    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for obj in four_corners_coordinate:
        if obj['cls'] in corners:
            result.append(obj)
            del corners[corners.index(obj['cls'])]
        else:
            print(f"Found duplicate {obj['cls']}! Deleted!")

    return result


def crop_card_final(img_path, model_path_4c, imgsz=640):
    #CROPPED_IMG_NAME = "img_" + str(count) + '.jpg'
    # load model
    model, names, stride, device, half = load_model(model_path_4c, imgsz)
    # read image from path and resize if it too large
    process_img = cv2.imread(img_path)
    resized_image = resize_if_too_large(process_img)

    start_time = time.time()

    # use model to predict four corners
    return_data = four_corners_detection(
        model, names, imgsz, stride, device, half, resized_image)
    four_corners_coordinate = check_duplicate_cls(return_data)
    if len(four_corners_coordinate) < 3:
        return
    #print("Cropping time: ", time.time()-start_time)

    # create temporary image for checking rotated image, if the image is rotated...
    # we will use this temporary image instead
    resized_image_2 = resize_if_too_large(process_img)
    # convert returned data to check wrong rotate image function's format
    four_corners_classes, four_corners_boxes = annotation_list_to_array(
        four_corners_coordinate)
    #print("Original image size: ", resized_image_2.shape)

    # Check rotation
    angle = check_wrong_rotate_image(resized_image_2, four_corners_coordinate)
    global cropped_img
    # if the image's angle is > 0 or < 0, we'll rotate it
    if angle:

        print(f"Wrong rotation direction! Need to rotate {angle}")
        im = Image.fromarray(resized_image_2)
        out = im.rotate(angle, expand=True)
        rotated_image = np.array(out)
        print("rotated image size: ", rotated_image.shape)
        new_four_corners_boxes = bboxes_rotation_final(
            resized_image_2, four_corners_boxes, angle)
        print("rotated bounding boxes: ", new_four_corners_boxes)

        # convert to yolov5 annotation format
        new_four_corners_coordinate = []

        for i in range(len(new_four_corners_boxes)):
            bbox = new_four_corners_boxes[i]
            cls = four_corners_classes[i]
            new_four_corners_coordinate.append({'bbox': bbox, 'cls': cls})

        # Draw bounding box rotation result
        #temp_rotated_image = np.copy(rotated_image)
        #temp_rotated_image = draw_bounding_box(temp_rotated_image, new_four_corners_coordinate)

        #cv2.imwrite(os.path.join(CORNERS_DETECTED_PATH, CROPPED_IMG_NAME), temp_rotated_image)

        # Original image need to be rotated too!
        im = Image.fromarray(process_img)
        out = im.rotate(angle, expand=True)
        rotated_process_img = np.array(out)

        cropped_img = crop_4corner(
            rotated_image, rotated_process_img, new_four_corners_coordinate)

    else:
        # Draw 4 corners detection result
        #resized_image = draw_bounding_box(resized_image, four_corners_coordinate)

        #cv2.imwrite(os.path.join(CORNERS_DETECTED_PATH, CROPPED_IMG_NAME), resized_image)
        #print("Rotated image boxes: ", new_four_corners_boxes)
        cropped_img = crop_4corner(
            resized_image_2, process_img, four_corners_coordinate)

    #cv2.imwrite(os.path.join(DEST_PATH, CROPPED_IMG_NAME), cropped_img)

    return cropped_img


# def generate_vietocr_data(images_path, labels_path, model_path_text, imgsz=640):
#     path = "../text_reg_dataset/"
#     images = [filename for filename in os.listdir(images_path)]
#     labels = [filename for filename in os.listdir(labels_path)]

#     model, names, stride, device, half = load_model(model_path_text, imgsz)

#     classes = ["cbeginning_date", "seri", "cid_no", "cname", "cdob", "cnationality",
#                "caddress", "issued_place", "issued_date", "cclass", "cexpires"
#                ]

#     image_count = 0
#     for i in range(len(images)):
#         print(f"reg {images[i]}")
#         f = open(os.path.join(labels_path, labels[i]), encoding="utf-8-sig")
#         label_dict = json.load(f)
#         img = cv2.imread(os.path.join(images_path, images[i]))
#         return_data = text_detection(
#             model, names, imgsz, stride, device, half, img)
#         text_count = 1
#         beginning_date_count = 0
#         address_count = 0

#         vietocr_annotations_file = open(os.path.join(
#             path, "annotations.txt"), "a+", encoding="utf8")

#         for obj in return_data:
#             if obj['cls'] not in classes:
#                 continue
#             xmin, ymin, xmax, ymax = obj['bbox']
#             text_img = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
#             print(obj['cls'])
#             text_name = f"img{image_count}_{text_count}_" + obj['cls'] + '.jpg'
#             text_images_path = os.path.join(path, "images")
#             cv2.imwrite(os.path.join(text_images_path, text_name), text_img)
#             if obj['cls'] == "cbeginning_date" and beginning_date_count == 0 and "beginning_date_2" in label_dict.keys():
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:] + "_2"] + "\n")
#                 beginning_date_count += 1
#             elif obj['cls'] == "cbeginning_date" and beginning_date_count == 0:
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:]] + "\n")
#                 beginning_date_count += 1
#             elif obj['cls'] == "cbeginning_date" and beginning_date_count == 1:
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:]] + "\n")
#             elif obj['cls'] == "caddress" and address_count == 0 and "address_2" in label_dict.keys():
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:] + "_2"] + "\n")
#                 address_count += 1
#             elif obj['cls'] == "caddress" and address_count == 0:
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:]] + "\n")
#                 address_count += 1
#             elif obj['cls'] == "caddress" and address_count == 1:
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:]] + "\n")
#             elif obj['cls'] == "issued_place" or obj['cls'] == "issued_date" or obj['cls'] == "seri":
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls']] + "\n")
#             else:
#                 vietocr_annotations_file.write(
#                     "images/" + text_name + "\t" + label_dict[obj['cls'][1:]] + "\n")
#             text_count += 1

#         vietocr_annotations_file.close()
#         image_count += 1


def text_recognition(img, text_detection_results, model_cfg):
    detector = Predictor(model_cfg)

    text_classes = ['cid_no', 'cname', 'cdob',
                    'cnationality', 'caddress',
                    'issued_place', 'issued_date'
                    'cclass', 'cexpires',
                    'classification', 'cbeginning_date',
                    'seri'
                    ]
    information_dict = {}

    for obj in text_detection_results:
        if obj['cls'] not in text_classes:
            continue
        #print(f"\nclass {obj['cls']}: ")
        xmin, ymin, xmax, ymax = obj['bbox']
        text_img_pil = Image.fromarray(img[int(ymin):int(ymax), int(xmin):int(xmax), :])
        reg_result = detector.predict(text_img_pil)

        if not obj['cls'] in information_dict.keys():
            information_dict[obj['cls']] = []
        information_dict[obj['cls']].append(reg_result)

    return information_dict


def text_reg_accuracy_evaluate(images_path, labels_path, model_path_text_detect, imgsz):
    labels = [filename for filename in os.listdir(labels_path)][:200]

    cid_no_predict = {}
    cname_predict = {}
    cdob_predict = {}
    cnationality_predict = {}
    caddress_predict = {}
    caddress_2_predict = {}
    issued_place_predict = {}
    issued_date_predict = {}
    cclass_predict = {}
    cexpires_predict = {}
    cbeginning_date_predict = {}
    cbeginning_date_2_predict = {}
    seri_predict = {}

    for i in range(len(labels)):
        f = open(os.path.join(labels_path, labels[i]), encoding="utf-8-sig")
        print("label " + labels[i] + ":")
        label_dict = json.load(f)
        img = cv2.imread(os.path.join(images_path, labels[i][:-5] + '.jpg'))
        information_dict, text_return_data = information_extraction_with_api(
            img, model_path_text_detect, imgsz)
        for key, value in label_dict.items():
            if key == "beginning_date":
                predict = (information_dict.get(
                    "cbeginning_date", [""])[0] == value)
                cbeginning_date_predict[labels[i][:-5]] = predict
            elif key == "beginning_date_2":
                try:
                    predict = (information_dict.get(
                        "cbeginning_date", ["", ""])[1] == value)
                except IndexError:
                    predict = False
                cbeginning_date_2_predict[labels[i][:-5]] = predict
            elif key == "seri":
                predict = (information_dict.get("seri", [""])[0] == value)
                seri_predict[labels[i][:-5]] = predict
            elif key == "id_no":
                predict = (information_dict.get("cid_no", [""])[0] == value)
                cid_no_predict[labels[i][:-5]] = predict
            elif key == "name":
                predict = (information_dict.get("cname", [""])[0] == value)
                cname_predict[labels[i][:-5]] = predict
            elif key == "dob":
                predict = (information_dict.get("cdob", [""])[0] == value)
                cdob_predict[labels[i][:-5]] = predict
            elif key == "nationality":
                predict = (information_dict.get(
                    "cnationality", [""])[0] == value)
                cnationality_predict[labels[i][:-5]] = predict
            elif key == "address":
                predict = (information_dict.get("caddress", [""])[0] == value)
                caddress_predict[labels[i][:-5]] = predict
            elif key == "address_2":
                try:
                    predict = (information_dict.get(
                        "caddress", ["", ""])[1] == value)
                except IndexError:
                    predict = False
                caddress_2_predict[labels[i][:-5]] = predict
            elif key == "class":
                predict = (information_dict.get("cclass", [""])[0] == value)
                cclass_predict[labels[i][:-5]] = predict
            elif key == "expires":
                predict = (information_dict.get("cexpires", [""])[0] == value)
                cexpires_predict[labels[i][:-5]] = predict
            elif key == "issued_place":
                predict = (information_dict.get(
                    "issued_place", [""])[0] == value)
                issued_place_predict[labels[i][:-5]] = predict
            elif key == "issued_date":
                predict = (information_dict.get(
                    "issued_date", [""])[0] == value)
                issued_date_predict[labels[i][:-5]] = predict

    print("id_no: ", cid_no_predict, sep="\n")
    print("name:", cname_predict, sep="\n")
    print("dob:", cdob_predict, sep="\n")
    print("nationality:", cnationality_predict, sep="\n")
    print("address:", caddress_predict, caddress_2_predict, sep="\n")
    print("issued date:", issued_date_predict, sep="\n")
    print("issued place:", issued_place_predict, sep="\n")
    print("class:", cclass_predict, sep="\n")
    print("expires:", cexpires_predict, sep="\n")
    print("beginning date:", cbeginning_date_predict,
          cbeginning_date_2_predict, sep="\n")
    print("seri:", seri_predict, sep="\n")

    id_no_samples = len(cid_no_predict.values())
    id_no_correct = sum(value != False for value in cid_no_predict.values())
    id_no_accuracy = id_no_correct / id_no_samples

    name_samples = len(cname_predict.values())
    name_correct = sum(value != False for value in cname_predict.values())
    name_accuracy = name_correct / name_samples

    dob_samples = len(cdob_predict.values())
    dob_correct = sum(value != False for value in cdob_predict.values())
    dob_accuracy = dob_correct / dob_samples

    nationality_samples = len(cnationality_predict.values())
    nationality_correct = sum(
        value != False for value in cnationality_predict.values())
    nationality_accuracy = nationality_correct / nationality_samples

    address_samples = len(caddress_predict.values()) + \
        len(caddress_2_predict.values())
    address_correct = sum(value != False for value in cdob_predict.values(
    )) + sum(value != False for value in caddress_2_predict.values())
    address_accuracy = address_correct / address_samples

    issued_date_samples = len(issued_date_predict.values())
    issued_date_correct = sum(
        value != False for value in issued_date_predict.values())
    issued_date_accuracy = issued_date_correct / issued_date_samples

    issued_place_samples = len(issued_place_predict.values())
    issued_place_correct = sum(
        value != False for value in issued_place_predict.values())
    issued_place_accuracy = issued_place_correct / issued_place_samples

    class_samples = len(cclass_predict.values())
    class_correct = sum(value != False for value in cclass_predict.values())
    class_accuracy = class_correct / class_samples

    expires_samples = len(cexpires_predict.values())
    expires_correct = sum(
        value != False for value in cexpires_predict.values())
    expires_accuracy = expires_correct / expires_samples

    beginning_date_samples = len(
        cbeginning_date_predict.values()) + len(cbeginning_date_2_predict.values())
    beginning_date_correct = sum(value != False for value in cbeginning_date_predict.values(
    )) + sum(value != False for value in cbeginning_date_2_predict.values())
    beginning_date_accuracy = beginning_date_correct / beginning_date_samples

    seri_samples = len(seri_predict.values())
    seri_correct = sum(value != False for value in seri_predict.values())
    seri_accuracy = seri_correct / seri_samples

    print(
        f"\nclass id number:\tsamples: {id_no_samples}\tcorrect: {id_no_correct}\taccuracy: {id_no_accuracy}")
    print(
        f"class name:\tsamples: {name_samples}\tcorrect: {name_correct}\taccuracy: {name_accuracy}")
    print(
        f"class dob:\tsamples: {dob_samples}\tcorrect: {dob_correct}\taccuracy: {dob_accuracy}")
    print(
        f"class nationality:\tsamples {nationality_samples}\tcorrect: {nationality_correct}\taccuracy: {nationality_accuracy}")
    print(
        f"class address:\tsamples: {address_samples}\tcorrect: {address_correct}\taccuracy: {address_accuracy}")
    print(
        f"class issued place:\tsamples: {issued_place_samples}\tcorrect: {issued_place_correct}\taccuracy: {issued_place_accuracy}")
    print(
        f"class issued date:\tsamples: {issued_date_samples}\tcorrect: {issued_date_correct}\taccuracy: {issued_date_accuracy}")
    print(
        f"class class:\tsamples: {class_samples}\tcorrect: {class_correct}\taccuracy: {class_accuracy}")
    print(
        f"class expires:\tsamples: {expires_samples}\tcorrect: {expires_correct}\taccuracy: {expires_accuracy}")
    print(
        f"class beginning date:\tsamples: {beginning_date_samples}\tcorrect: {beginning_date_correct}\taccuracy: {beginning_date_accuracy}")
    print(
        f"class seri:\tsamples: {seri_samples}\tcorrect: {seri_correct}\taccuracy: {seri_accuracy}")

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-i', '--image_path', type=str, required=False,
                    help='path to single input image')
    ap.add_argument('-i_sz', '--image_size', type=int, required=False,
                    help='input image size', default=640)
    ap.add_argument('-fc_w', '--four_corners_weight', type=str, required=False,
                    help='path to four corner detection weight',
                    default="./weights/4c_det_weight.pt")
    ap.add_argument('-td_w', '--text_det_weight', type=str, required=False,
                    help='path to text detection weight',
                    default="./weights/text_det_weight.pt")
    ap.add_argument('-tr_w', '--text_reg_weight', type=str, required=False,
                    help='path to text recognition weight',
                    default="./weights/text_reg_weight.pth")
    args = ap.parse_args()

    imgsz = args.image_size

    FC_DETECTION_WEIGHT_PATH = args.four_corners_weight
    TEXT_DETECTION_WEIGHT_PATH = args.text_det_weight
    TEXT_RECOGNITION_WEIGHT_PATH = args.text_reg_weight
    #DEST_PATH = "../data/data-text/cropped_front/"

    CORNERS_DETECTED_PATH = "./cropped_card/"

    #text_reg_img_raw_path = "../data/data-reg/data-reg-raw-final/"
    #text_reg_labels_path = "../data/data-reg/data-reg-labels-final/"

    #images = [filename for filename in os.listdir(FOLDER_IMAGES)]

    if args.image_path:
        # FIND AND CROP THE CARD

        print("1. CROP CARD\n")
        IMG_PATH = args.image_path
        start = time.time()
        cropped_img = crop_card_final(
            IMG_PATH, 
            FC_DETECTION_WEIGHT_PATH, 
            imgsz
        )
        print(f"\n--time exec: {time.time() - start}\n")

        cv2.imwrite(os.path.join(CORNERS_DETECTED_PATH,
                                 "cropped.jpg"), cropped_img)
        
        # TEXT DETECTION
        print("2. INFORMATION FIELDS DETECTION\n")
        text_detection_model, text_detection_names, text_detection_stride, text_detection_device, text_detection_half = load_model(TEXT_DETECTION_WEIGHT_PATH, imgsz)

        start = time.time()
        text_detection_results = text_detection(
            text_detection_model, 
            text_detection_names, 
            imgsz, 
            text_detection_stride, 
            text_detection_device, 
            text_detection_half, 
            cropped_img
        )

        print(f"--time exec: {time.time() - start}\n")
        
        # TEXT RECOGNITION
        print("3. INFORMATION FIELDS RECOGNITION\n")

        cfg = Cfg.load_config_from_name('vgg_transformer')
        cfg['weights'] = TEXT_RECOGNITION_WEIGHT_PATH
        cfg['cnn']['pretrained'] = False
        cfg['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg['predictor']['beamsearch'] = False

        start = time.time()
        information_dict = text_recognition(cropped_img, text_detection_results, cfg)
        print(f"--time exec: {time.time() - start}\n")

        print(f"{5 * '='}RESULTS{5 * '='}")

        for key, name in information_dict.items():
            print(f"{key}: {name[0]}")
    #text_reg_accuracy_evaluate(text_reg_img_raw_path, text_reg_labels_path, model_path_text_detect, imgsz)

    #generate_vietocr_data(text_reg_img_raw_path, text_reg_labels_path, model_path_text_detect, imgsz)

    #visualize_crop(cropped_img, text_return_data)

if __name__ == "__main__":
    main()
