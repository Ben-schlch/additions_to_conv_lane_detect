import argparse
import glob
import json
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import scipy
from scipy import interpolate
from tqdm import tqdm


def draw(im, line, idx, show=False):
    '''
    Generate the segmentation label according to json annotation
    '''
    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int(line_x[0]), int(line_y[0]))
    if show:
        cv2.putText(im, str(idx), (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0, (int(line_x[i + 1]), int(line_y[i + 1])), (idx,), thickness=16)
        pt0 = (int(line_x[i + 1]), int(line_y[i + 1]))


def euclidean_distance(x1: int, x2: int, y1: int, y2: int) -> float:
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_polyline_length(polyline: List[Tuple[int, int]]) -> float:
    distances = []
    for i in range(len(polyline) - 1):
        current_line = polyline[i]
        next_line = polyline[i + 1]
        distances.append(euclidean_distance(current_line[0], next_line[0], current_line[1], next_line[1]))

    return sum(distances)


def spline_interpolation(x: np.ndarray, y: np.ndarray, num_interpolations: int, degree: int = 3) -> List[List[int]]:
    tck = interpolate.splprep([x, y], k=degree, s=20)[0]
    u3 = np.linspace(0, 1, num_interpolations, endpoint=True)
    x_interpolated, y_interpolated = interpolate.splev(u3, tck)
    try:
        x_interpolated = list(map(int, x_interpolated))
        y_interpolated = list(map(int, y_interpolated))
    except Exception as e:
        print(x)
        print(y)
        print(num_interpolations)
        print(degree)
        print(x_interpolated)
        print(y_interpolated)

        raise e

    return [x_interpolated, y_interpolated]


def interpolate_lanes(lane: List) -> List:
    interpolations = []
    for polyline in lane:
        polyline = [x for x in polyline if x[0] is not None]

        seen = set()
        polyline_cleaned = []
        for coord in polyline:
            t = tuple(coord)
            if t not in seen:
                polyline_cleaned.append(coord)
            seen.add(t)

        if len(polyline_cleaned) < 2:
            continue

        x_coords, y_coords = zip(*polyline_cleaned)
        polyline_length = compute_polyline_length(polyline_cleaned)

        if len(x_coords) > 3 and len(y_coords) > 3:
            lane_points_interpolated = list(zip(*spline_interpolation(x_coords, y_coords, int(polyline_length))))
        elif len(x_coords) == 3 and len(y_coords) == 3:
            lane_points_interpolated = list(zip(*spline_interpolation(x_coords, y_coords, int(polyline_length), 2)))
        elif len(x_coords) == 2 and len(y_coords) == 2:
            lane_points_interpolated = list(zip(*spline_interpolation(x_coords, y_coords, int(polyline_length), 1)))
        else:
            continue

        interpolations.append(lane_points_interpolated)

    return interpolations


def get_sections_2(lanes):
    interpolated = np.vstack(interpolate_lanes(lanes))
    the_anno_row_anchor = np.array(range(0, 1544, 10))
    col_for_row_anchor = np.full((len(the_anno_row_anchor)), -99999)

    for idx, anchor_row in enumerate(the_anno_row_anchor):
        anchor_col_list = interpolated[np.where(interpolated[:, 1] == anchor_row)]
        if len(anchor_col_list) == 0:
            continue

        col_for_row_anchor[idx] = np.mean(anchor_col_list, axis=0)[0]

    return np.vstack([col_for_row_anchor, the_anno_row_anchor]).T.tolist()


def get_smartrollerz_dataset_2(root: str, labels_list: List[str]) -> dict:
    with open(os.path.join(root, labels_list[0]), 'r') as file:  # TODO: load all label files
        labels: dict = json.load(file)

    return labels


def generate_segmentation_and_train_list_2(root: str, labels: dict):
    train_gt_file = open(os.path.join(root, 'train_gt.txt'), 'w')
    cache_dict = {}

    for label in labels:
        all_points = []
        image_path = ""
        lanes = None
        for image, lane_label in label.items():
            image_path = image
            lanes = lane_label

        if image_path == "" or lanes is None:
            continue

        all_points.append(get_sections(lanes['left_lane']))
        all_points.append(get_sections(lanes['center_lane']))
        all_points.append(get_sections(lanes['right_lane']))

        # draw_annotated_lanes_on_image(os.path.join(root, image_path), all_points)

        image_seg_path = image_path[:-4] + '_seg' + image_path[-4:]
        image_seg = np.zeros((1544, 2064), dtype=np.uint8)

        cv2.imwrite(os.path.join(root, image_seg_path), image_seg)

        cache_dict[image_path] = all_points
        train_gt_file.write(image_path + ' ' + image_seg_path + ' ' + ' 1 1 1' + '\n')

    train_gt_file.close()
    with open(os.path.join(root, 'smartrollerz_anno_cache.json'), 'w') as f:
        json.dump(cache_dict, f)


############################################################################


def load_labels(path: str) -> dict:
    if os.path.isdir(path):
        label_filenames = glob.glob(path + '/**/*.json', recursive=True)
    elif path[:-5] == ".json":
        label_filenames = [path]
    else:
        label_filenames = []

    if len(label_filenames) == 0:
        raise FileNotFoundError(f"No json files were found under {path}!")

    labels_dict = {}

    for filename in label_filenames:
        print(f"Importing labels from {filename}.")
        with open(filename, 'r') as file:
            imported_labels: dict = json.load(file)
            labels_dict = {**labels_dict, **imported_labels}

    return labels_dict


def get_sections(lanes):
    interpolated = np.vstack(interpolate_lanes(lanes))
    the_anno_row_anchor = np.array(range(300, 1544 - 600, 10))
    col_for_row_anchor = np.full((len(the_anno_row_anchor)), -99999)

    for idx, anchor_row in enumerate(the_anno_row_anchor):
        anchor_col_list = interpolated[np.where(interpolated[:, 1] == anchor_row)]
        if len(anchor_col_list) == 0:
            continue

        col_for_row_anchor[idx] = np.mean(anchor_col_list, axis=0)[0]

    return np.vstack([col_for_row_anchor, the_anno_row_anchor]).T.tolist()


def generate_segmentation_and_train_list(root: str, labels_dict: dict):
    train_gt_file = open(os.path.join(root, 'labels', 'train_gt.txt'), 'w')
    cache_dict = {}

    for image_path, image_dict in tqdm(labels_dict.items()):

        image_seg_path = image_path[:-4] + '_seg' + image_path[-4:]
        image_seg = np.zeros((1544, 2064), dtype=np.uint8)  # TODO: create proper segmentation image
        cv2.imwrite(os.path.join(root, image_seg_path), image_seg)

        # the_anno_row_anchor = np.array(range(0, 1544, 10))
        the_anno_row_anchor = np.array(range(300, 1544 - 600, 10))
        col_for_row_anchor = np.full((len(the_anno_row_anchor)), -99999)
        empty_lane = np.vstack([col_for_row_anchor, the_anno_row_anchor]).T.tolist()

        cache_dict[image_path] = []
        if 'left_lane' in image_dict['lanes'].keys():
            cache_dict[image_path].append(get_sections(image_dict['lanes']['left_lane']))
        else:
            cache_dict[image_path].append(empty_lane)
        if 'center_lane' in image_dict['lanes'].keys():
            cache_dict[image_path].append(get_sections(image_dict['lanes']['center_lane']))
        else:
            cache_dict[image_path].append(empty_lane)
        if 'right_lane' in image_dict['lanes'].keys():
            cache_dict[image_path].append(get_sections(image_dict['lanes']['right_lane']))
        else:
            cache_dict[image_path].append(empty_lane)

        train_gt_file.write(
            f"{image_path} {image_seg_path} "
            f"{1 if 'left_lane' in image_dict['lanes'].keys() else 0} "
            f"{1 if 'center_lane' in image_dict['lanes'].keys() else 0} "
            f"{1 if 'right_lane' in image_dict['lanes'].keys() else 0}"
            f"\n"
        )

    train_gt_file.close()
    with open(os.path.join(root, 'labels', 'smartrollerz_anno_cache.json'), 'w') as file:
        json.dump(cache_dict, file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Smartrollerz dataset')
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()

    # Training dataset
    labels_v1 = load_labels(path=os.path.join(args.root, 'labels', 'v1', 'train'))
    labels_v2 = load_labels(path=os.path.join(args.root, 'labels', 'v2', 'train'))
    labels = {**labels_v1, **labels_v2}
    generate_segmentation_and_train_list(args.root, labels)

    # Test dataset
    test_labels_v1 = load_labels(path=os.path.join(args.root, 'labels', 'v1', 'test'))
    test_labels_v2 = load_labels(path=os.path.join(args.root, 'labels', 'v2', 'test'))
    test_labels = {**test_labels_v1, **test_labels_v2}

    with open(os.path.join(args.root, 'labels', 'test.txt'), 'w') as file:
        for image_path, _ in test_labels.items():
            file.write(image_path + '\n')
