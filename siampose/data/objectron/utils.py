import typing

import cv2 as cv
import numpy as np

import siampose.data.objectron.dataset.graphics
import siampose.data.utils


def display_debug_frames(
        sample_frame: typing.Dict,
):
    cv.imshow("raw", sample_frame["IMAGE"])
    assert (sample_frame["POINT_NUM"] == 9).all() and sample_frame["INSTANCE_NUM"] > 0
    points_2d = sample_frame["POINT_2D"].reshape((sample_frame["INSTANCE_NUM"], 9, 3))
    image = np.copy(sample_frame["IMAGE"])
    for instance_idx in range(sample_frame["INSTANCE_NUM"]):
        image = siampose.data.objectron.dataset.graphics.draw_annotation_on_image(
            image, points_2d[instance_idx, :, :], [9])
        cv.circle(image, tuple(sample_frame["CENTROID_2D_IM"][instance_idx]), 3, (255, 255, 255), -1)
    cv.imshow("2d", image)
    points_3d = sample_frame["POINT_3D"].reshape((sample_frame["INSTANCE_NUM"], 9, 3))
    view_matrix = sample_frame["VIEW_MATRIX"].reshape((4, 4))
    proj_matrix = sample_frame["PROJECTION_MATRIX"].reshape((4, 4))
    im_width, im_height = image.shape[1], image.shape[0]
    image = np.copy(sample_frame["IMAGE"])
    for instance_idx in range(sample_frame["INSTANCE_NUM"]):
        curr_points_3d = points_3d[instance_idx]
        points_2d_new = siampose.data.utils.project_points(
            curr_points_3d, proj_matrix, view_matrix, im_width, im_height)
        for point_id in range(points_2d_new.shape[0]):
            cv.circle(image, (points_2d_new[point_id, 0], points_2d_new[point_id, 1]), 10,
                      (0, 255, 0), -1)
    cv.imshow("3d", image)
    cv.waitKey(0)


def sort_sample_keypoints(
        sample: typing.Dict[typing.AnyStr, typing.Any],
):
    # assume bboxes are always laying (almost) flat on ground plane
    # top-bottom-left-right from 2D
    # far/close from 3D w/ ground plane
    # ... the idea is to get more stable bboxes while giving up on the exact kpt placement
    # ... IoU does not care about keypoint order
    required_keys = [
        "POINTS", "POINT_3D", "VIEW_MATRIX", "PROJECTION_MATRIX",
        "PLANE_CENTER", "PLANE_NORMAL",
    ]
    nb_frames_per_tuple = len(sample["POINTS"])
    assert all([k in sample and len(sample[k]) == nb_frames_per_tuple for k in required_keys])
    # good = output quality flag, will be set to false if anything goes wrong in this sample
    good = True
    # by definition, keypoint#0 is the object center, but the other ones might be sketchy
    for frame_idx in range(nb_frames_per_tuple):
        kpts3d = sample["POINT_3D"][frame_idx].reshape((-1, 3))
        center2d, center3d = sample["POINTS"][frame_idx][0], kpts3d[0]
        kpts2d, kpts3d = sample["POINTS"][frame_idx][1:], kpts3d[1:]
        plane_normal, plane_center = sample["PLANE_NORMAL"][frame_idx], sample["PLANE_CENTER"][frame_idx]
        plane_d = np.dot(-plane_normal, plane_center)
        kpts_plane_dists = np.asarray([
            siampose.data.utils.distance_between_point_and_plane(
                pt[0], pt[1], pt[2],
                plane_normal[0], plane_normal[1], plane_normal[2], plane_d,
            ) for pt in kpts3d
        ])
        # if all goes well and the assumption holds, we should have 4 points pretty close to 0 dist
        good = good and (sum([np.isclose(dist, 0., atol=0.005) for dist in kpts_plane_dists]) == 4)
        sort_idxs = np.argsort(kpts_plane_dists)
        kpts2d, kpts3d = kpts2d[sort_idxs], kpts3d[sort_idxs]
        # the non-ground keypoints should also all be at the same height (distance)
        ngkpts_dists = kpts_plane_dists[sort_idxs][4:]
        good = good and np.allclose(ngkpts_dists, ngkpts_dists.max(), atol=0.005)
        # next, check cam dist to identify close/far points
        kpts_cam_dists = np.asarray([np.linalg.norm(pt) for pt in kpts3d])
        sort_idxs_bottom, sort_idxs_top = \
            np.argsort(kpts_cam_dists[:4]), np.argsort(kpts_cam_dists[4:])
        kpts2d[:4], kpts3d[:4] = kpts2d[:4][sort_idxs_bottom], kpts3d[:4][sort_idxs_bottom]
        kpts2d[4:], kpts3d[4:] = kpts2d[4:][sort_idxs_top], kpts3d[4:][sort_idxs_top]
        # finally, check 2D left/right assignments and swap pairs if necessary
        if kpts2d[1][0] > kpts2d[2][0]:
            kpts2d[1], kpts2d[2] = kpts2d[2], kpts2d[1].copy()
            kpts3d[1], kpts3d[2] = kpts3d[2], kpts3d[1].copy()
        if kpts2d[5][0] > kpts2d[6][0]:
            kpts2d[5], kpts2d[6] = kpts2d[6], kpts2d[5].copy()
            kpts3d[5], kpts3d[6] = kpts3d[6], kpts3d[5].copy()
        # as post-proc validation: make sure top-down pairs are all aligned the same way
        pair_vecs = kpts3d[4:] - kpts3d[:4]
        good = good and np.allclose(pair_vecs, pair_vecs.max(axis=0), atol=0.005)
        # we're done, just reassign the sorted kpts back to the sample
        sample["POINT_3D"][frame_idx] = np.concatenate([center3d.reshape((1, 3)), kpts3d]).flatten()
        sample["POINTS"][frame_idx] = np.concatenate([center2d.reshape((1, 2)), kpts2d])
    return good
