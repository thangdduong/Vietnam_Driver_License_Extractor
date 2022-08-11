import cv2
import numpy as np
from PIL import Image

def get_center_point(coordinate_dict):
    """
    Convert xmin, ymin, xmax, ymax format to x_center, y_center, width, height format
    -----------------------
    input:
    1. coordinate_dict: list of dictionar of bounding box

    output: dictionary of bounding boxes {'class name': bbox}
    """
    di = dict()

    for obj in coordinate_dict:
        xmin, ymin, xmax, ymax = obj['bbox']
        x_center = int((xmin + xmax) / 2)
        y_center = int((ymin + ymax) / 2) 
        di[obj['cls']] = [x_center, y_center, int(xmax - xmin), int(ymax - ymin)]
    
    return di

def annotation_list_to_array(coordinate_list):
    """
    Convert detected bbox coordinates (from list to seperated classes, boxes arrays)
    -----------------------
    input:
    1. coordinate_dict: list of dictionar of bounding box

    output: list of class names and array of bounding box
    """
    classes = []
    list_with_all_boxes = []
    for obj in coordinate_list:
        classes.append(obj['cls'])
        list_with_all_boxes.append(obj['bbox'])

    return classes, np.array(list_with_all_boxes)

def get_corners(bboxes):
    """
    Convert xmin ymin xmax ymax (topleft and bottomright coordinate) to x1 y1 x2 y2 x3 y3 x4 y4 (4 corners coordinate)
    -------------------------------
    input:
    1. bboxes: list of xmin ymin xmax ymax bbox

    output: an array of x1 y1 x2 y2 x3 y3 x4 y4 bbox
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)

    x2 = x1 + width
    y2 = y1 

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

    return corners

def rotate_box(corners, angle, cx, cy, h, w):
    """
    Rotate bbox by given angle
    -------------------------
    input:
    1. corners: an array of x1 y1 x2 y2 x3 y3 x4 y4 bbox
    2. angle: an int value of desired rotation angle
    3. cx: x value of center point of image
    4. cy: y value of center point of image
    5. h: height of image
    6. w: width of image

    output: an array of rotated x1 y1 x2 y2 x3 y3 x4 y4 bbox
    """
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T

    calculated = calculated.reshape(-1,8)

    return calculated

def get_enclosing_box(corners):
    """
    Convert x1 y1 x2 y2 x3 y3 x4 y4 bbox to xmin ymin xmax ymax bbox
    -------------------------
    input:
    1. corners: an array of x1 y1 x2 y2 x3 y3 x4 y4 bbox

    output: an array of xmin ymin xmax ymax bbox
    """
    x = corners[:, [0,2,4,6]]
    y = corners[:, [1,3,5,7]]

    xmin = np.min(x, 1).reshape(-1, 1)
    ymin = np.min(y, 1).reshape(-1, 1)
    xmax = np.max(x, 1).reshape(-1, 1)
    ymax = np.max(y, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:,8:]))

    return final

def bboxes_rotation_final(img, bboxes, angle):
    """
    Find the rotated bbox of corners given an angle.
    -----------------------
    input:
    1. img: an image
    2. bboxes: 

    output: list of rotated bounding box
    """
    w, h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    corners = get_corners(bboxes)
    corners = np.hstack((corners, bboxes[:,4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bboxes = get_enclosing_box(corners)

    new_bboxes = new_bboxes.astype('int32')

    return new_bboxes.tolist()

def check_wrong_rotate_image(image, coordinate_dict):
    """
    Find the angle of the image.
    ------------------------
    input:
    1. image: an image
    2. coordinate_dict: 4 corners bounding box dictionary

    output: an int value indicates the angle of image (90, -90, 180)
    """
    angle = 0
    h, w = image.shape[:2]
    h_center, w_center = h//2, w//2
    centered_dict = get_center_point(coordinate_dict)

    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for corner in corners:
        if corner in centered_dict.keys():
            continue
        centered_dict[corner] = [-1, -1, -1 , -1]

    if (centered_dict["bottom_left"][0] in range(w_center + 1) and centered_dict["bottom_right"][0] in range(w_center + 1)) or (centered_dict["bottom_left"][0] in range(w_center + 1) and centered_dict["top_left"][0] in range(w_center, w + 1)) or (centered_dict["top_right"][0] in range(w_center, w + 1) and centered_dict["top_left"][0] in range(w_center, w + 1)) or (centered_dict["bottom_right"][0] in range(w_center + 1) and centered_dict["top_right"][0] in range(w_center, w + 1)):
        angle = 90
    elif (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["bottom_right"][0] in range(w_center, w + 1)) or (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["top_left"][0] in range(w_center + 1)) or (centered_dict["top_left"][0] in range(w_center + 1) and centered_dict["top_right"][0] in range(w_center + 1)) or (centered_dict["top_right"][0] in range(w_center + 1) and centered_dict["bottom_right"][0] in range(w_center, w + 1)):
        angle = -90
    elif (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["bottom_right"][0] in range(w_center + 1)) or (centered_dict["bottom_left"][0] in range(w_center, w + 1) and centered_dict["top_left"][0] in range(w_center, w + 1)) or (centered_dict["top_right"][0] in range(w_center + 1) and centered_dict["bottom_right"][0] in range(w_center + 1)) or (centered_dict["top_left"][0] in range(w_center, w + 1) and centered_dict["top_right"][0] in range(w_center + 1)):
        angle = 180

    return angle

def check_duplicate_cls(four_corners_coordinate):
    """
    Check if there are duplicate classes in dict of given dict of corners bounding box. If exists, remove one the duplicates.
    --------------------
    input:
    1. four_corners_coordinate: dictionary of corner bounding boxes

    output: list of {bbox, cls}
    """
    result = []
    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for obj in four_corners_coordinate:
        if obj['cls'] in corners:
            result.append(obj)
            del corners[corners.index(obj['cls'])]
        else:
            print(f"Found duplicate {obj['cls']}! Deleted!")
    
    return result

def rotate_to_standard(img, return_data):
    """
    Rotate image and its corner bounding boxes to standard position.
    -------------------------
    input:
    1. img: an image
    2. return_data: a list of dictionary of bounding boxes (output from yolov5 corners detection)

    output: rotated image and dictionary of rotated bounding boxes
    """
    four_corners_coordinate = check_duplicate_cls(return_data)
    if len(four_corners_coordinate) < 3:
        print("There are not enough corners to rotate")

        return None, None
    
    # convert returned data to check wrong rotate image function's format
    four_corners_classes, four_corners_boxes = annotation_list_to_array(four_corners_coordinate)

    # Check if image is in wrong position
    angle = check_wrong_rotate_image(img, four_corners_coordinate)

    # if the image's angle is > 0 or < 0, we'll rotate it
    if angle:
        # rotate image 
        im = Image.fromarray(img)
        out = im.rotate(angle, expand=True)
        rotated_image = np.array(out)

        # find rotated bounding boxes
        new_four_corners_boxes = bboxes_rotation_final(img, four_corners_boxes, angle)

        # convert to yolov5 annotation format
        new_four_corners_coordinate = []
        
        for i in range(len(new_four_corners_boxes)):
            bbox = new_four_corners_boxes[i]
            cls = four_corners_classes[i]
            new_four_corners_coordinate.append({'bbox': bbox, 'cls': cls})

        return rotated_image, new_four_corners_coordinate

    else: 
        return img, return_data
