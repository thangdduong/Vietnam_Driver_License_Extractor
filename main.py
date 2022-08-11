import time
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=False,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=False,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=False,
                help='path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def resize_if_too_large(input_img):
    h, w = input_img.shape[:2]
    #resize input image if its size is too large
    if w > 750:
        new_w = 750
        new_h = 750/w * h
        input_img = cv2.resize(input_img, (int(new_w), int(new_h)), cv2.INTER_AREA)
    
    return input_img

def calc_distance(point_a, point_b):
    return math.sqrt( (point_b[0] - point_a[0])**2 + (point_b[1] - point_a[1])**2 )

def get_center_point(coordinate_dict):
    di = dict()
    # corner_offset = {'top_left': [-30, -30], 'top_right': [30, -30], 'bottom_left': [-30, 30], 'bottom_right': [30, 30]}
    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key]
        print(f"{key}: {coordinate_dict[key]}")
        x_center = ((xmin + xmax) / 2)
        y_center = ((ymin + ymax) / 2) 
        # if key == 'top_left':
        #     x_center = xmin
        #     y_center = ymin
        # elif key == 'top_right':
        #     x_center = xmax
        #     y_center = ymin
        # elif key == 'bottom_left':
        #     x_center = xmin
        #     y_center = ymax
        # else:
        #     x_center = xmax
        #     y_center = ymax
        di[key] = [x_center, y_center]

    return di

def four_corners_detection():
    image = cv2.imread(args.image)
    image = resize_if_too_large(image)
    # cv2.imshow('image', image)
    # cv2.waitKey()

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open("classes/4c-classes.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet("weights/4c-832x832-cfg.weights", "cfg/4c-832x832.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (832, 832), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    coordinate_dict = {}
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        coordinate_dict[classes[class_ids[i]]] = [round(x), round(y), round(x + w), round(y + h)]
        draw_prediction(image, classes, COLORS, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    cv2.imwrite("four-corners-detection.jpg", image)
    #cv2.imshow("object detection", image)


    end = time.time()
    print("YOLO Execution time: " + str(end-start))


    # cv2.waitKey()

    # cv2.imwrite("object-detection.jpg", image)
    # cv2.destroyAllWindows()
    return coordinate_dict

### Crop 4 corners

def perspective_transform(image, x_ratio, y_ratio, corners):
    # Order points in clockwise order
    #ordered_corners = order_corner_points(corners)
    #top_l, top_r, bottom_r, bottom_l = ordered_corners
    top_l, top_r, bottom_r, bottom_l = corners

    h, w = image.shape[:-1]
    # offsets = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    offset = 30
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
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(corners, dtype="float32")
    print("ordered_corners", ordered_corners)
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

    topleft_center = [coordinate_dict["top_left"][:2]]
    topright_center = [coordinate_dict["top_right"][:2]]
    botright_center = [coordinate_dict["bottom_right"][:2]]
    botleft_center = [coordinate_dict["bottom_left"][:2]]
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
            tl_x = topright_center[0][0] - calc_distance(botright_center[0], botleft_center[0]) 
            tl_y = botleft_center[0][1] - calc_distance(botright_center[0], topright_center[0])
            topleft_center = [[tl_x, tl_y]]
        elif len(topright_center) == 0:
            tr_x = topleft_center[0][0] + calc_distance(botright_center[0], botleft_center[0])
            tr_y = botright_center[0][1] - calc_distance(botleft_center[0], topleft_center[0])
            topright_center = [[tr_x, tr_y]]
        elif len(botright_center) == 0:
            br_x = botleft_center[0][0] + calc_distance(topright_center[0], topleft_center[0])
            br_y = topright_center[0][1] + calc_distance(botleft_center[0], topleft_center[0])
            botright_center = [[br_x, br_y]]
        elif len(botleft_center) == 0:
            bl_x = botright_center[0][0] - calc_distance(topright_center[0], topleft_center[0])
            bl_y = topleft_center[0][1] + calc_distance(botright_center[0], topright_center[0])
            botleft_center = [[bl_x, bl_y]]

    x_ratio = original_img.shape[1]/ resized_img.shape[1]
    y_ratio = original_img.shape[0]/ resized_img.shape[0]
#    print(x_ratio, y_ratio)

    corners_center = (topleft_center[0], topright_center[0], botright_center[0], botleft_center[0])
#    print(corners_center)
    transformed = perspective_transform(original_img, x_ratio, y_ratio, corners_center)
    #print(transformed.shape)
    cv2.imshow("cropped_img", transformed)
    cv2.imwrite('cropped_img.jpg', transformed)
    cv2.waitKey()

    
    return transformed

def crop_image(coordinate_dict):
    image = cv2.imread(args.image)
    resized_image = resize_if_too_large(image)
    cropped_image = crop_4corner(resized_image, image, coordinate_dict)

    return cropped_image

def text_detection(image):
    image = resize_if_too_large(image)
    # cv2.imshow('image', image)
    # cv2.waitKey()

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open("classes/text-classes.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet("weights/text-603x603-cfg.weights", "cfg/text-603x603.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    coordinate_dict = {}
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        coordinate_dict[classes[class_ids[i]]] = [round(x), round(y), round(x + w), round(y + h)]
        draw_prediction(image, classes, COLORS, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    cv2.imwrite("text-detection.jpg", image)
    #cv2.imshow("object detection", image)
    end = time.time()
    print("YOLO Execution time: " + str(end-start))


    # cv2.waitKey()

    # cv2.imwrite("object-detection.jpg", image)
    # cv2.destroyAllWindows()
    return coordinate_dict


def main():
    four_corners_coordinate_dict = four_corners_detection()
    print(four_corners_coordinate_dict)
    cropped_img = crop_image(four_corners_coordinate_dict)
    text_coordinate_dict = text_detection(cropped_img)
    print(text_coordinate_dict)


if __name__ == '__main__':
    main()