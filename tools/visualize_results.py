import os
import copy
import numpy as np
import cv2
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import torch

import mmcv
from mmdet3d.core.bbox import LiDARInstance3DBoxes

f_scores = open("states_scores_3donly_2.log", "w")

cam_lidar_pair = torch.load("/adafuse/cam_lidar_pair/cam_lidar_pair_sample784.pth")
sample_id = 0


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
    k: int = 0,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)

        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]
        scores = scores[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]
        scores = scores[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)

        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )

                label_text = (
                    f"{name} {int(scores[index]*100)}%" if scores is not None else name
                )
                label_position = (
                    int(coords[index, 0, 0]),
                    int(coords[index, 0, 1] - 10),
                )
                cv2.putText(
                    canvas,
                    label_text,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_bbox(data=None, data_name=None, outputs=None, cfg=None, mode="pred"):
    global cam_lidar_pair, sample_id
    out_dir = "/adafuse/viz_adpative_010724"

    print("=" * 50)
    print(
        f"[!CHECK! @ visualize_results.py] output directory for visualized results: {out_dir}"
    )
    print("=" * 50)

    metas = data["img_metas"][0].data[0][0]

    # cam_lidar_pair[pts_name] = {}
    # cam_lidar_pair[pts_name]['filename'] = metas['filename']
    # cam_lidar_pair[pts_name]['lidar2img'] = metas['lidar2img']
    # torch.save(cam_lidar_pair, os.path.join("/futr3d/cam_lidar_pair", f"cam_lidar_pair_sample{sample_id}.pth"))
    # sample_id += 1

    bbox_classes = None
    bbox_score = 0.3
    map_score = 0.5

    if mode == "gt" and "gt_bboxes_3d" in data:
        bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        labels = data["gt_labels_3d"].data[0][0].numpy()

        if bbox_classes is not None:
            indices = np.isin(labels, bbox_classes)
            bboxes = bboxes[indices]
            labels = labels[indices]

        # bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    elif mode == "pred" and "boxes_3d" in outputs:
        bboxes = outputs["boxes_3d"].tensor.numpy()
        scores = outputs["scores_3d"].numpy()
        labels = outputs["labels_3d"].numpy()

        if bbox_classes is not None:
            indices = np.isin(labels, bbox_classes)
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        if bbox_score is not None:
            indices = scores >= bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

            print(
                f"length: {len(scores)} sum: {sum(scores)} mean: {sum(scores)/len(scores)} min: {min(scores)} max: {max(scores)}"
            )
            f_scores.write(
                f"length: {len(scores)} sum: {sum(scores)} mean: {sum(scores)/len(scores)} min: {min(scores)} max: {max(scores)}\n"
            )

        # bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    else:
        bboxes = None
        labels = None

    # if mode == "gt" and "gt_masks_bev" in data:
    #     masks = data["gt_masks_bev"].data[0].numpy()
    #     masks = masks.astype(np.bool)
    # elif mode == "pred" and "masks_bev" in outputs:
    #     masks = outputs["masks_bev"].numpy()
    #     masks = masks >= map_score
    # else:
    #     masks = None

    if "img" in data:
        for k, image_path in enumerate(metas["filename"]):
            image = mmcv.imread(image_path)
            visualize_camera(
                os.path.join(out_dir, f"camera-{k}", f"{data_name}.png"),
                image,
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                transform=metas["lidar2img"][k],
                classes=cfg.class_names,
               k=k,
            )
    if "points" in data:
        pts_name = metas["pts_filename"]
        print (f"pts_name: {pts_name}")
        #check if pts_name is key of cam_lidar_pair
        if pts_name not in cam_lidar_pair:
            print ("pts_name not in cam_lidar_pair")
            return
        for k, image_path in enumerate(cam_lidar_pair[pts_name]["filename"]):
            image = mmcv.imread(image_path)
            visualize_camera(
                os.path.join(out_dir, f"camera-{k}", f"{data_name}.png"),
                image,
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                transform=cam_lidar_pair[pts_name]["lidar2img"][k],
                classes=cfg.class_names,
                k=k,
            )

        lidar = data["points"][0].data[0][0].numpy()
        visualize_lidar(
            os.path.join(out_dir, "lidar", f"{data_name}.png"),
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=cfg.class_names,
        )
