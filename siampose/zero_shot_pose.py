#from notebooks.opencv_install.opencv_contrib.modules.dnn_objdetect.scripts.pascal_preprocess import rescale
import PIL
import numpy
from numpy.core.fromnumeric import argmax
import pandas as pd
import open3d as o3d
import random
import math
import re
import h5py
import io
import multiprocessing
#from multiprocessing import Pool
from multiprocessing import get_context

#multiprocessing.set_start_method('spawn') #Fork does not play nice with h5py!
#Reference: https://pythonspeed.com/articles/python-multiprocessing/

import numpy as np
import cupy as cp
import os

from tqdm import tqdm

from deco import concurrent, synchronized
from tqdm.utils import RE_ANSI

import siampose.data.objectron.data_transforms
import siampose.data.objectron.sequence_parser
import siampose.data.utils
import siampose.geometry as geo

from siampose.data.objectron.dataset import iou
from siampose.data.objectron.dataset import box

from google.protobuf import text_format
# The annotations are stored in protocol buffer format. 
from siampose.data.objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from siampose.data.objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol

from PIL import Image, ImageDraw
EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane = bottom
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

BOTTOM_POINTS = [1, 2, 6, 5]

use_cupy = True


def get_center_ajust(result_bbox, projected_center):
    cx_result = (result_bbox[0] + result_bbox[2])/2
    cy_result = (result_bbox[1] + result_bbox[3])/2
    adjust_x = projected_center[0] - cx_result
    adjust_y = projected_center[1] - cy_result
    adjust_x_rel = adjust_x / (result_bbox[2]-result_bbox[0])
    adjust_y_rel = adjust_y / (result_bbox[3]-result_bbox[1])
    return adjust_x_rel, adjust_y_rel

def estimate_object_center_in_query_image(query_intrinsics, query_bbox, points_2d_px_result):
    cx = (query_bbox[0] + query_bbox[2])/2
    cy = (query_bbox[1] + query_bbox[3])/2
    result_bbox = geo.get_bbox(points_2d_px_result, 360, 480) 
    projected_center = points_2d_px_result[0]
    adjust_x_rel, adjust_y_rel = get_center_ajust(result_bbox, projected_center)
    adjust_x_px = adjust_x_rel * (query_bbox[2]-query_bbox[0])
    adjust_y_px = adjust_y_rel * (query_bbox[3]-query_bbox[1])
    cx = cx + adjust_x_px
    cy = cy + adjust_y_px
    b = (cx - query_intrinsics[1,2])/query_intrinsics[1,1]
    a = (cy - query_intrinsics[0,2])/query_intrinsics[0,0]
    return cx, cy, a, b




def align_box_with_plane(points3d, plane_normal_query, plane_normal_result):
    box_rotation = geo.rotation_matrix_from_vectors(plane_normal_query, plane_normal_result)
    #plane_normal_query, plane_normal_result, plane_center_query, plane_center_result
    points_3d_axis = np.dot(points3d-points3d[0],box_rotation)+points3d[0]
    return points_3d_axis

def get_scale_factor(query_bbox, points3d_scaled, intrinsics, width=360, height=480):
    points2d_px_result = geo.project_3d_to_2d(points3d_scaled, intrinsics)
    #points2d_px_query = geo.project_3d_to_2d(points_3d_query, intrinsics)
    result_bbox = geo.get_bbox(points2d_px_result, width, height)
    #query_bbox = geo.get_bbox(points2d_px_query, width, height)
    scale = geo.get_bbox_area(query_bbox) / geo.get_bbox_area(result_bbox)
    return scale

def get_smooth_scale_factor(query_bbox, points3d_scaled, intrinsics, alpha):
    factor = get_scale_factor(query_bbox, points3d_scaled, intrinsics)    
    return (alpha+factor-1)/alpha

def get_bounding_box(idx, match_idx, experiment, adjust_scale=False):
    points_2d_result, points_3d_result = experiment.get_points(match_idx, train=True)
    points_2d_px_result = geo.points_2d_to_points2d_px(points_2d_result, 360, 480)
    points_2d_query, _ = experiment.get_points(idx, train=False)
    points_2d_px_query = geo.points_2d_to_points2d_px(points_2d_query, 360, 480)
    plane_center_query, plane_normal_query= experiment.get_plane(idx, train=False)
    plane_center_result, plane_normal_result = experiment.get_plane(match_idx, train=True)
    result_bbox = geo.get_bbox(points_2d_px_result, 360, 480) 
    #query_camera = experiment.get_camera(idx, train=False)
    query_intrinsics = experiment.get_intrinsics(idx, train=False)
    query_intrinsics = geo.scale_intrinsics(query_intrinsics, 0.25,0.25)
    result_bbox = geo.get_bbox(points_2d_px_result, 360, 480)
    query_bbox = geo.get_bbox(points_2d_px_query, 360, 480)
    cx, cy, a, b = estimate_object_center_in_query_image(query_intrinsics, query_bbox, points_2d_px_result)
    center_ray = np.array([a,b,-1])
    points_3d_axis = align_box_with_plane(points_3d_result, plane_normal_query, plane_normal_result)
    obj_radius = geo.get_dist_from_plane(plane_normal_result, plane_center_result, points_3d_axis[0])
    points_3d_result_snapped = geo.snap_to_plane(points_3d_axis, plane_normal_query, plane_center_query, center_ray, obj_radius)
    if adjust_scale:
        scale = get_smooth_scale_factor(query_bbox, points_3d_result_snapped, query_intrinsics, 2)
        points3d_scaled = points_3d_result_snapped
        for i in range(0,4):
            #print(scale, obj_radius)
            obj_radius = geo.get_dist_from_plane(plane_normal_query, plane_center_query, points3d_scaled[0])
            points3d_scaled = geo.snap_to_plane(geo.scale_3d_bbox(points3d_scaled, scale),
                                                plane_normal_query, plane_center_query, center_ray, obj_radius = obj_radius*scale)
            scale = get_smooth_scale_factor(query_bbox, points3d_scaled, query_intrinsics, 2)
            #print(i, get_iou_between_bbox(np.array(points_3d_query), np.array(points3d_scaled)))
        return points3d_scaled
    return points_3d_result_snapped


def create_objectron_bbox_from_points(points, color=(0,0,0)):
    assert len(color)==3
    #EDGES = (
    #    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    #    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    #    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    #)
    pcd = o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, pcd, EDGES)
    colors = np.array(color*len(lineset.lines)).reshape(len(lineset.lines),3)
    lineset.colors=o3d.utility.Vector3dVector(colors)
    return lineset



BOTTOM_POINTS = [1, 2, 6, 5]




def get_match_aligned_points(idx:int, match_idx:int, info_df: pd.DataFrame, 
            train_info_df: pd.DataFrame,
            ground_truth=False):
    """Get 3D IoU for a specific query image, using the kth neighbor.

    Returns:
        [type]: [description]
    """
    assert type(match_idx)==int
    points2d_train, points3d_train = get_points(train_info_df, match_idx)
    points2d_valid, points3d_valid = get_points(info_df, idx)
        
    valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
    train_image = Image.open(train_info_df.iloc[match_idx]["filepath_full"])
    points2d_valid_px = geo.points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
    points2d_train_px = geo.points_2d_to_points2d_px(points2d_train, train_image.width, train_image.height)
    valid_bbox = geo.get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
    train_bbox = geo.get_bbox(points2d_train_px, train_image.width, train_image.height)
    points3d_train_rotated, _ = geo.align_with_bbox_3d(points3d_train, train_bbox, valid_bbox)
    if ground_truth:
        return np.array(points3d_train_rotated), np.array(points3d_valid)
    else:
        return np.array(points3d_train_rotated), np.array(points3d_train)




def get_objectron_bbox_colors():
    bbox_point_colors = np.zeros((9,3))
    
    for bottom_point in BOTTOM_POINTS:
        bbox_point_colors[bottom_point]=np.array([1,0,0])
    bbox_point_colors[0] = np.array([0,0,1])
    return o3d.utility.Vector3dVector(bbox_point_colors)  










def fix_idx(idx):
    if type(idx) is not int and type(idx) is not numpy.ndarray:
        idx = int(idx) #Just make sure this is the right 
    return idx




def visualize(points3d_query1, points3d_results1, points3d_results2):
    """Small visualisation helper. Interface will change!

    """

    pcd_results1 = o3d.geometry.PointCloud()
    pcd_results2 = o3d.geometry.PointCloud()
    #points3d_train_results1 = points3d_results1
    #points3d_train_rotated, points3d_train_aligned = align_with_bbox_3d(points3d_train, train_bbox, valid_bbox)
    pcd_results1.points = o3d.utility.Vector3dVector(points3d_results1)
    pcd_results2.points = o3d.utility.Vector3dVector(points3d_results2)
    
    #pcd_train_aligned.points = o3d.utility.Vector3dVector(points3d_train_aligned)

    pcd_valid = o3d.geometry.PointCloud()
    pcd_valid.points = o3d.utility.Vector3dVector(points3d_query1)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    bbox_3d_results1 = pcd_results1.get_oriented_bounding_box()
    bbox_3d_results2 = pcd_results2.get_oriented_bounding_box()
    bbox_3d_valid = pcd_valid.get_oriented_bounding_box()
    #bbox_3d.create_from_point_cloud_poisson(pcd_train)
    bbox_3d_results1.color = [0,0,255]
    bbox_3d_results2.color = [255,0,0]
    vis.add_geometry(bbox_3d_results1)
    vis.add_geometry(bbox_3d_results2)
    vis.add_geometry(bbox_3d_valid)
    vis.add_geometry(pcd_results1)
    vis.add_geometry(pcd_results2)
    vis.add_geometry(pcd_valid)

    print(bbox_3d_results1.extent, bbox_3d_results1.center)
    print(bbox_3d_valid.extent, bbox_3d_valid.center)
    

    vis.run()
    ctr = vis.get_view_control()
    pinhole_parameters = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    #pinhole_parameters.intrinsic
    #vis.draw_geometries()
    return 


def get_match_snapped_points(idx, match_idx,  info_df, train_info_df, ground_truth=False, rescale=True, align_axis=True):
    assert type(match_idx)==int
    plane_center, plane_normal = get_plane(info_df, idx)
    points2d, points3d = get_points(info_df,idx)
    points3d_result_rotated, points3d_result = get_match_aligned_points(idx, match_idx, info_df, train_info_df, ground_truth = ground_truth)
    snapped, intersect = geo.snap_box_to_plane(points3d_result_rotated, plane_normal, plane_center, align_axis=align_axis)
    

    result = snapped
    if rescale:
        points2d_result, _ = get_points(train_info_df,match_idx)
        #points3d_result = np.array(points3d_result)
        camera = get_camera(info_df, idx)
        intrinsics = get_intrinsics(camera)
        intrinsics = geo.scale_intrinsics(intrinsics, 0.25, 0.25)
        points2d_px = geo.project_3d_to_2d(snapped, intrinsics)
        dest_bbox = geo.get_bbox(points2d_px, 360, 480)

        points2d_valid, points3d_valid = get_points(info_df, idx)

        valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
        points2d_valid_px = geo.points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
        valid_bbox = geo.get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
        scale_x, scale_y = geo.get_scale_factors(dest_bbox, valid_bbox)
        scale_factor = (scale_x + scale_y)/2
        
        #print(scale_factor)
        fixed_point = snapped[0]
        snapped_rotated = snapped.copy()
        #scale_factor = 1.2
        snapped_rotated=(snapped_rotated-fixed_point)*scale_factor+fixed_point
        snapped_rotated, intersect=geo.snap_box_to_plane(snapped_rotated, plane_normal, plane_center)
        result = snapped_rotated

    return result, points3d_result

def get_iou_between_bbox(points_train, points_valid):
    try:
        v_rotated = np.array(points_train)
        v_valid = np.array(points_valid)
        w_rotated = box.Box(vertices=v_rotated)
        w_valid = box.Box(vertices=v_valid)
        loss = iou.IoU(w_rotated, w_valid)
        return float(loss.iou())
    except:
        print("Error computing IoU!")
        return 0

def get_iou(idx:int, embeddings: np.array, info_df: pd.DataFrame, 
            train_embeddings:np.array, train_info_df: pd.DataFrame,
            symmetric=False, rescale=False,
            k=0, show=False, compute_aligned=False, ground_plane=True,
            align_axis=True
            ):
    """Get 3D IoU for a specific query image, using the kth neighbor.

    Args:
        idx (int): Query index in meta dataframe.
        embeddings (np.array): Query embeddings.
        info_df (pd.DataFrame): Query embeddings metadata.
        train_embeddings (np.array): Trainin set embeddings. Will be used for lookup.
        train_info_df (pd.DataFrame): Training set embeddings metadata.
        k (int, optional): kth neighbor to use as result.. Defaults to 0.
        show (bool, optional): Visualize the result in 3D. Defaults to False.

    Returns:
        [type]: [description]
    """
    #@concurrent
    match_idx = find_match_idx(idx, embeddings, train_embeddings,k)
    if ground_plane:
        points3d_processed, points3d_valid = get_match_snapped_points(idx, match_idx, info_df, train_info_df, ground_truth=True, rescale=rescale, align_axis=align_axis)
    else:
        points3d_processed, points3d_valid = get_match_aligned_points(idx, match_idx, info_df, train_info_df, ground_truth=True)
 
    #if show:
    #    visualize(points3d_valid, points3d_train_rotated, points3d_train_aligned)
    iou_value = get_iou_between_bbox(points3d_valid, points3d_processed)
    best_iou = iou_value
    if symmetric:
        for angle in np.arange(5,360,5):
            theta = (angle/180)*np.pi
            pivoted = geo.rotate_bbox_around_its_center(points3d_processed, theta)
            iou_at_theta = get_iou_between_bbox(pivoted, points3d_valid)
            if iou_at_theta > best_iou:
                best_iou = iou_at_theta
        #_get_iou.wait()
    return best_iou , match_idx



def get_iou_rotated(points3d_processed, points3d_valid, initial_iou):
    """Get the IoU metric for a symmetric object by rotating it around its axis.

    Args:
        points3d_processed (np.array): Result Bounding box 
        points3d_valid (np.array): Query Bounding box
        initial_iou (float): Initial IoU when the object is not rotated.

    Returns:
        float: Best IoU obtained while rotating the object around its axis.
    """
    best_iou = initial_iou
    for angle in np.arange(5,360,5):
        theta = (angle/180)*np.pi
        pivoted = geo.rotate_bbox_around_its_center(points3d_processed, theta)
        iou_at_theta = get_iou_between_bbox(pivoted, points3d_valid)
        if iou_at_theta > best_iou:
            best_iou = iou_at_theta
    return best_iou

def find_match_idx(idx, query_embeddings: np.ndarray, train_embeddings:np.ndarray, k=0, score=False):
    """Find the nearest neighbor of a single query embedding in the test set in the training set.

    Args:
        idx (int): Index in the test set.
        query_embeddings (np.ndarray): Test set embeddings.
        train_embeddings (np.ndarray): Train set embeddings.
        k (int, optional): kth neighbor to use. Defaults to 0.

    Returns:
        int: Match index in the train set.
    """
    global use_cupy
    idx = fix_idx(idx)
    lib = cp
    if not use_cupy:
        lib = np
    
    similarity = lib.dot(query_embeddings[idx].T,train_embeddings)
    if k != 0:
        best_matches = lib.argsort(-similarity)
        match_idx = best_matches[k]
    else:
        match_idx = lib.argmin(-similarity)
    if not score:
        return int(match_idx)
    else:
        return int(match_idx), float(similarity[match_idx])

def find_all_match_idx(query_embeddings: np.ndarray, train_embeddings:np.ndarray, k=0):
    """Find all matches for the test set in the training set at the sametime, using cupy.

    This is solely for optimisation purpose in order to get the code to run faster on machines
    with GPU.

    Args:
        query_embeddings (np.ndarray): Test set embeddings.
        train_embeddings (np.ndarray): Train set embeddings.
        k (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: The case where "k!=0" is not yet implemeted.

    Returns:
        [type]: [description]
    """
    global use_cupy
    print("Using GPU to compute matches!")

    if k != 0:
        #best_matches = cp.argsort(-cp.dot(query_embeddings.T,train_embeddings))
        #match_idx = best_matches[k]
        raise ValueError("The case where k is not 0 must be implemented.")
    else:
        match_idxs = []
        query_chunk_size = 1024
        train_chunk_size = 65536*2
        for i in tqdm(range(0, math.ceil(len(query_embeddings)/query_chunk_size))):
            query_start = i*query_chunk_size
            query_end = query_start + query_chunk_size
            if query_end > len(query_embeddings):
                query_end=len(query_embeddings)
            cuda_query_embeddings = cp.asarray(query_embeddings[query_start:query_end])

            matches = []
            scores = []
            best_match_idx_chunk_score = np.zeros((query_end-query_start,1))
            best_match_idx_chunk = np.zeros((query_end-query_start,1), dtype=np.uint64)
            for j in range(0,math.ceil(train_embeddings.shape[1]/train_chunk_size)):
                train_start = j*train_chunk_size
                train_end = train_start + train_chunk_size
                if train_end > train_embeddings.shape[1]:
                    train_end=train_embeddings.shape[1]
                cuda_train_embeddings = cp.asarray(train_embeddings[:,train_start:train_end])
                similarity = cp.dot(cuda_query_embeddings,cuda_train_embeddings)
                match_idx_chunk = cp.argmax(similarity,axis=1).get()
                similarity=similarity.get()
                match_idx_chunk_score = np.take_along_axis(similarity,np.expand_dims(match_idx_chunk,axis=1),axis=1)
                match_idx_chunk+=train_start
                best_match_idx_chunk = np.where(match_idx_chunk_score>best_match_idx_chunk_score, np.expand_dims(match_idx_chunk, axis=1), best_match_idx_chunk).astype(np.uint64)
                best_match_idx_chunk_score = np.where(match_idx_chunk_score>best_match_idx_chunk_score,match_idx_chunk_score,best_match_idx_chunk_score)
                
                #if use_cupy:
                #match_idx_chunk=match_idx_chunk.get()

                matches.append(match_idx_chunk)
            match_idxs+=best_match_idx_chunk.squeeze().tolist()
        return match_idxs





def describe_intrinsics(intrinsics: np.array):
    """Describe the intrinsics matrix in their common names.

    Args:
        intrinsics (np.array): Intrinsics
    """
    alpha_x = intrinsics[0,0]
    alpha_y = intrinsics[1,1]
    center_x = intrinsics[0,2]
    center_y = intrinsics[1,2]
    return float(alpha_x), float(alpha_y), float(center_x), float(center_y)  

#embeddings = None
#info_df = None
#train_embeddings = None
#train_info_df = None
experiment = None

args = None
all_match_idxs = None
import argparse
import time


def get_iou_mp(idx:int #, symmetric=False, rescale=False,
            #k=0, show=False, compute_aligned=False, ground_plane=True,
            ):
    """Get 3D IoU for a specific query image, using the kth neighbor.

    Args:
        idx (int): Query index in meta dataframe.
        embeddings (np.array): Query embeddings.
        info_df (pd.DataFrame): Query embeddings metadata.
        train_embeddings (np.array): Trainin set embeddings. Will be used for lookup.
        train_info_df (pd.DataFrame): Training set embeddings metadata.
        k (int, optional): kth neighbor to use as result.. Defaults to 0.
        show (bool, optional): Visualize the result in 3D. Defaults to False.

    Returns:
        [type]: [description]
    """
    
    global args, experiment, all_match_idxs
    experiment.load_hdf5_file() #Just make sure the hdf5 file is not loader prior to forking.
    category = experiment.get_category(idx, train=False)

    #all_match_idxs = find_match_idx(idx, embeddings, train_embeddings,0)
    match_idx=all_match_idxs[idx]
    if args.random_bbox:
        match_idx = random.randint(0,len(experiment.train_info_df)-1)
    if args.random_bbox_same:
        match_idx = random.sample(list(experiment.train_info_df.query(f"category=='{category}'").index),1)
        match_idx = match_idx[0]

    #if args.legacy:
    #    if args.ground_plane:
    #        points3d_processed, points3d_valid = get_match_snapped_points(idx, match_idx, experiment.info_df, train_info_df, ground_truth=True, rescale=args.rescale, align_axis=not args.no_align_axis)
    #    else:
    #        points3d_processed, points3d_valid = get_match_aligned_points(idx, match_idx, experiment.info_df, train_info_df, ground_truth=True)
    #else:
    _, points3d_valid = experiment.get_points(idx, train=False)
    points3d_processed = get_bounding_box(idx, match_idx, experiment, adjust_scale=True)
    #if show:
    #    visualize(points3d_valid, points3d_train_rotated, points3d_train_aligned)
    iou_value = get_iou_between_bbox(points3d_valid, points3d_processed)
    best_iou = iou_value
    if args.symmetric:        
        if category=="cup" or category=="bottle":
            best_iou = get_iou_rotated(points3d_processed, points3d_valid, best_iou)
        #_get_iou.wait()
    return best_iou , idx, match_idx

class ExperimentHandlerFile():
    def __init__(self, experiment:str=None, test=False):
        self.info_df = None
        self.train_info_df = None
        self.hdf5_dataset = None
        self.hdf5_test_dataset = None
        self.test= test
        if experiment:
            self.read_experiment(experiment, test)
        self.source_image_width = 1440
        self.source_image_height = 1920

    def read_experiment(self, experiment: str, test: bool):
        """Read the output of a SimSiam experiment folder. (Embeddings and metadata)

        Args:
            experiment (str): Output folder location 

        Returns:
            tuple: validation embeddings, validation meta data, train embeddings, train meta data
        """
        global use_cupy
        load_fn = cp.load
        if not use_cupy:
            load_fn = np.load
        
        info_prefix = ""
        if test:
            info_prefix="test_"
            print("Loading experiment with test set instead of validation set embeddings!")
        embeddings = load_fn(f"{experiment}/{info_prefix}embeddings.npy")
        if embeddings.shape [1]>embeddings.shape [0]:
            embeddings=embeddings.T
        info = numpy.load(f"{experiment}/{info_prefix}info.npy")
        assert info.shape[0]==2
        train_new_filename = f"{experiment}/train_embeddings.npy"
        train_old_filename = f"{experiment}/training_embeddings.npy"
        if os.path.exists(train_new_filename):
            train_embeddings = load_fn(train_new_filename)
        else:
            train_embeddings = load_fn(train_old_filename)
        train_info = numpy.load(f"{experiment}/train_info.npy")
        assert train_info.shape[0]==2

            #lines = content
        self.init_experiment(embeddings, info, train_embeddings, train_info)
        try:
            if not self.test:
                with open(f"{experiment}/train_sequences.txt","r") as f:
                    train_sequences_t = f.read().split("\n")
                    valid_sequences_g = sorted(list((self.info_df["sequence_uid"].unique())))
                    for valid_sequence in valid_sequences_g:
                        if valid_sequence in train_sequences_t:
                            raise ValueError(f"The validation sequence {valid_sequence} was part of the training set!")
        except FileNotFoundError:
            print(f"Warning, unable to find {experiment}/train_sequences.txt.\n We will not be able to make sure there is no overlap between training and validation!")

    def load_hdf5_file(self):
        if self.hdf5_dataset is None:
            self.hdf5_dataset = h5py.File('/home/raphael/datasets/objectron/extract_s5_raw.hdf5','r')
        if self.test and self.hdf5_test_dataset is None:
            self.hdf5_test_dataset = h5py.File('/home/raphael/datasets/objectron/extract_s5_raw_test.hdf5','r')

    def init_experiment(self, embeddings, info, train_embeddings, train_info):
        info_df = pd.DataFrame(info.T)
        train_info_df = pd.DataFrame(train_info.T)
        info_df_columns = {0:"category_id",1:"uid"}
        train_info_df = train_info_df.rename(columns=info_df_columns)
        info_df = info_df.rename(columns=info_df_columns)
        if "hdf5" in train_info_df.iloc[0][1]:
            self.mode = "hdf5"
            self.hdf5_dataset = None
        else:
            self.mode = "raw"
        self.parse_info_df(info_df)
        self.parse_info_df(train_info_df, subset="train")
        assert train_embeddings.shape[0] == embeddings.shape[1]
        self.embeddings = embeddings
        self.train_embeddings = train_embeddings
        self.info_df = info_df
        self.train_info_df = train_info_df



    def _pick_hdf5_dataset(self, train):
        if not self.test:
            return self.hdf5_dataset
        if not train and self.test:
            return self.hdf5_test_dataset
        if train:
            return self.hdf5_dataset
        raise ValueError("We should get here!")

    def get_points(self, idx: int, train=False):
        if self.mode=="raw":
            return self._raw_get_points(idx, train)
        elif self.mode=="hdf5":
            category, sequence, image_id  = self._hdf5_parse_uid(idx,train)
            hdf5_dataset = self._pick_hdf5_dataset(train)
            image_idx = list(hdf5_dataset[category][sequence]["IMAGE_ID"]).index(image_id)
            return hdf5_dataset[category][sequence]["POINT_2D"][image_idx].reshape(9,3), hdf5_dataset[category][sequence]["POINT_3D"][image_idx].reshape(9,3)
        else:
            raise ValueError(f"Mode not supported: {self.mode}")

    def get_image(self,idx, train=False):
        df = self._get_df(train)
        if self.mode=="hdf5":
            category, sequence, image_id  = self._hdf5_parse_uid(idx,train)
            hdf5_dataset = self._pick_hdf5_dataset(train)
            image_idx = list(hdf5_dataset[category][sequence]["IMAGE_ID"]).index(image_id)
            jpeg_data = hdf5_dataset[category][sequence]["IMAGE"][image_idx]
            #image = Image.frombytes('RGBA', (128,128), jpeg_data, 'raw')
            image = Image.open(io.BytesIO(jpeg_data))
            #img= cv2.imdecode(self.hdf5_dataset[category][sequence]["IMAGE"][image_idx],cv2.IMREAD_ANYCOLOR)
            return image
        else:
            raise ValueError("To be implemented!")

    def _get_df(self, train):
        if train:
            df = self.train_info_df
        else:
            df = self.info_df
        return df

    def _hdf5_parse_uid(self, idx, train=False):
        df = self._get_df(train)
        uid = df.iloc[idx]["uid"]
        m = re.match("hdf5_(\w+)/(\d+)_(\d+)", uid)
        category, sequence, image_id = m[1],m[2], int(m[3])
        return category, sequence, image_id 

    def _raw_get_points(self, idx: int, train=False):
        """Get 2d and 3d points for a specific frame.

        Args:
            df (pd.DataFrame): Embeddings meta data frame.
            idx (int): Index of the frame in the dataframe.

        Returns:
            tuple: 2d points, 3d points relative to camera.
        """
        if train:
            df = self.train_info_df
        else:
            df = self.info_df

        idx = fix_idx(idx)
        sequence_annotations = self._raw_get_sequence_annotations(df.iloc[idx]["category"], df.iloc[idx]["batch_number"], df.iloc[idx]["sequence_number"])
        frame = int(df.iloc[idx]["frame"])
        object_id = int(df.iloc[idx]["object_id"])
        keypoints = sequence_annotations.frame_annotations[frame].annotations[object_id].keypoints
        points_2d = []
        points_3d = []
        for keypoint in keypoints:
            point_2d = (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth)
            point_3d = (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z)
            points_2d.append(point_2d)
            points_3d.append(point_3d)
        return np.array(points_2d), np.array(points_3d)

    def _raw_get_sequence_annotations(self, category: str, batch_number: str, sequence_number: str, annotations_path="/home/raphael/datasets/objectron/annotations"):
        """Get annotation data for a video sequence.

        Args:
            category (str): Category in the objectron dataset.
            batch_number (str): Batch number. 
            sequence_number (str): Sequence number. 
            annotations_path (str, optional): Location of the Objectron annotation files. Defaults to "/home/raphael/datasets/objectron/annotations".

        Returns:
            annotations: Google's annotations in their format.
        """
        sequence = annotation_protocol.Sequence()
        annotation_file=f"{annotations_path}/{category}/{batch_number}/{sequence_number}.pbdata"
        with open(annotation_file, 'rb') as pb:
            sequence.ParseFromString(pb.read())
        return sequence

    def get_plane(self, idx: int, train=False):
        if self.mode=="raw":
            return self._raw_get_plane(idx, train)
        elif self.mode=="hdf5":
            category, sequence, image_id  = self._hdf5_parse_uid(idx,train)
            hdf5_dataset = self._pick_hdf5_dataset(train)
            image_idx = list(hdf5_dataset[category][sequence]["IMAGE_ID"]).index(image_id)
            return hdf5_dataset[category][sequence]["PLANE_CENTER"][image_idx], hdf5_dataset[category][sequence]["PLANE_NORMAL"][image_idx]
        else:
            raise ValueError(f"Mode not supported: {self.mode}")
 
    def _raw_get_plane(self, idx: int, train=False):
        """Get object plane for a specific camera frame.

        Args:
            df (pd.DataFrame): Embeddings meta data frame.
            idx (int): Index of the frame in the dataframe.

        Returns:
            tuple: 2d points, 3d points relative to camera.
        """
        if train:
            df = self.train_info_df
        else:
            df = self.info_df
        idx = int(idx)
        sequence_annotations = self._raw_get_sequence_annotations(df.iloc[idx]["category"], df.iloc[idx]["batch_number"], df.iloc[idx]["sequence_number"])
        frame = int(df.iloc[idx]["frame"])

        plane_center = sequence_annotations.frame_annotations[frame].plane_center
        plane_normal = sequence_annotations.frame_annotations[frame].plane_normal

        return np.array(plane_center), np.array(plane_normal)

    
    def _raw_get_camera(self, idx:int, train=False):
        """Get camera information for a specific trame. 

        Args:
            df (pd.DataFrame): Embeddings meta data frame.
            idx (int): Index of the frame in the dataframe.

        Returns:
            object: Google's camera information for the frame.
        """
        if train:
            df = self.train_info_df
        else:
            df = self.info_df
        idx = fix_idx(idx)
        sequence_annotations = self._raw_get_sequence_annotations(df.iloc[idx]["category"], df.iloc[idx]["batch_number"], df.iloc[idx]["sequence_number"])
        frame = int(df.iloc[idx]["frame"])
        frame_annotation = sequence_annotations.frame_annotations[frame]
        return frame_annotation.camera

    def _raw_get_intrinsics(self, idx, train=False):
        camera = self._raw_get_camera(idx, train)
        return np.array(camera.intrinsics).reshape(3,3)

    def get_intrinsics(self, idx, train=False):
        if self.mode=="raw":
            return self._raw_get_intrinsics(idx, train)
        elif self.mode=="hdf5":
            category, sequence, image_id  = self._hdf5_parse_uid(idx,train)
            hdf5_dataset = self._pick_hdf5_dataset(train)
            image_idx = list(hdf5_dataset[category][sequence]["IMAGE_ID"]).index(image_id)
            return hdf5_dataset[category][sequence]["INTRINSIC_MATRIX"][image_idx].reshape(3,3)
        else:
            raise ValueError(f"Mode not supported: {self.mode}")

    @staticmethod
    def _get_subset(info_df, embeddings, ratio):
        if ratio==1:
            return info_df, embeddings
        #info_df["video_uid"]=info_df["category"]+"_"+info_df["video_id"]
        video_uids = list(info_df["sequence_uid"].unique())
        videos_uids_subset = random.sample(video_uids, int(len(video_uids)*ratio))
        print(f"Using a subset of {len(videos_uids_subset)} out of {len(video_uids)} total sequences for evaluation.")
    
        info_df_subset =info_df[info_df["sequence_uid"].isin(videos_uids_subset)]
        embeddings_subset=embeddings[:,list(info_df_subset.index)]
        info_df_subset = info_df_subset.reset_index()
        return info_df_subset, embeddings_subset

    def set_trainsubset(self, ratio):
        train_info_df, train_embeddings = self._get_subset(self.train_info_df, self.train_embeddings, ratio)
        self.train_info_df = train_info_df
        self.train_embeddings = train_embeddings

    def get_category(self, idx, train=False):
        df = self._get_df(train)
        if self.mode=="raw":
            return df.iloc[idx]["category"]
        elif self.mode=="hdf5":
            category, _, _  = self._hdf5_parse_uid(idx,train)
            return category
        else:
            raise ValueError(f"Mode not supported: {self.mode}") 

    def get_sequence_uid(self, idx, train=False):
        df = self._get_df(train)
        return df.iloc[idx]["sequence_uid"]
            

    def parse_info_df(self, info_df, subset="valid"):
        """Parses an embedding meta-data dataframe. (The output of SimSiam)

        Args:
            info_df (pd.DataFrame): Pandas Dataframe, as output by the "read experiment" function.
            subset (str, optional): Which subset to use. Can be "train", "valid" or test. Defaults to "valid".
        """
        info_df["category"]=info_df["uid"].str.extract("(.*?)-")
        info_df["sequence_uid"]=info_df["uid"].str.extract("(batch\-\d+_\d+_\d+)")
        info_df["frame"]=info_df["uid"].str.extract("-(\d+)$")
        info_df["video_id"]=info_df["uid"].str.extract("(batch\-\d+_\d+)")
        info_df["object_id"]=info_df["uid"].str.extract("batch\-\d+_\d+_(\d+)")
        info_df["batch_number"]=info_df["uid"].str.extract("(batch\-\d+)")
        info_df["sequence_number"]=info_df["uid"].str.extract("batch\-\d+_(\d+)_\d+")
        info_df["filepath"]=f"/home/raphael/datasets/objectron/96x96/{subset}/" + info_df["category"] +"/" + info_df["sequence_uid"] +"." + info_df["frame"] + ".jpg"
        info_df["filepath_full"]="/home/raphael/datasets/objectron/640x480_full/" + info_df["category"] +"/" + info_df["sequence_uid"] +"." + info_df["frame"] + ".jpg"
        if self.mode=="hdf5":
            info_df["sequence_uid"] = info_df["uid"].str.extract("hdf5_(\w+/\d+)_")

def main():
    global args, experiment, ground_plane, symmetric, rescale, all_match_idxs, use_cupy
    """Command line tool for evaluating zero shot pose estimation
    on objectron."""
    parser = argparse.ArgumentParser(description='Command line tool for evaluating zero shot pose estimation on objectron.')
    parser.add_argument("experiment", type=str, help="Experiment folder location. i.e. ../SimSiam/outputs/objectron_96x96_experiment_synchflip_next")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of samples for 3D IoU evaluation")
    parser.add_argument("--ground_plane", default=True, help="If enabled, snap to ground plane")
    parser.add_argument("--iou_t", default=0.5, help="IoU threshold required to consider a positive match")
    parser.add_argument("--symmetric", action="store_true",help="Rotate symetric objects (cups, bottles) and keep maximum IoU.")
    parser.add_argument("--rescale", action="store_true",help="Rescale 3d bounding box")
    parser.add_argument("--random_bbox", action="store_true", help="Fit a randomly selected bbox instead of the nearest neighbor")
    parser.add_argument("--random_bbox_same", action="store_true", help="Fit a randomly selected bbox from same category instead of the nearest neighbor")
    parser.add_argument("--trainset-ratio", type=float, default=1, help="Ratio of the training set sequences used for inference")
    parser.add_argument("--single_thread", action="store_true", help="Disable multithreading.")
    parser.add_argument("--cpu", action="store_true", help="Disable cuda accelaration.")
    parser.add_argument("--no_align_axis", action="store_true", help="Don't to to align axis with ground plane.")
    parser.add_argument("--legacy", action="store_true", help="Deprecated legacy evalution mode")
    parser.add_argument("--test", action="store_true", help="Evaluate on test set embeddings")
    args = parser.parse_args()
    symmetric = args.symmetric
    rescale = args.rescale
    
    
    
    if args.cpu:
        use_cupy=False

    experiment = ExperimentHandlerFile(args.experiment, args.test)

    if args.trainset_ratio > 0 and args.trainset_ratio <= 1:
        experiment.set_trainsubset(args.trainset_ratio)
    else:
        raise ValueError("Training set ratio must be between 0 and 1!")

    all_match_idxs = find_all_match_idx(experiment.embeddings, experiment.train_embeddings, 0)
    #get_iou_mp(1)
    if args.subset_size < len(experiment.info_df):
        subset = random.sample(range(0,len(experiment.info_df)),args.subset_size)
    else:
        print(f"Evaluating on all samples size subset size ({args.subset_size}) is larger that test set size ({len(experiment.info_df)}).")
        subset = list(range(0,len(experiment.info_df)))
        random.shuffle(subset)

    ious = {}
    results = []
    ious_aligned = []
    threshold = args.iou_t
    valid_count = 0
    get_iou_params = []

    for idx in subset:
        #params = (idx, args.symmetric, args.rescale)
        get_iou_params.append(idx)
    if not args.single_thread:
        with get_context("fork").Pool(4) as p: 
            results = list(tqdm(p.imap(get_iou_mp, get_iou_params), total=len(subset)))
    else:
        for idx in tqdm(get_iou_params):
            results.append(get_iou_mp(idx))

    for iou, idx, match_idx in results:
        category = experiment.get_category(idx, train=False)
        if not category in ious:
            ious[category]=[]
        if iou > threshold:
            valid_count+=1
        ious[category].append(float(iou))

    ious = pd.DataFrame.from_dict(ious, orient='index').T
    ious.to_csv("raw_results.txt")
    print(f"{'category' : <10}\tmean iou\tmedian iou\tAP at iou_t")
    for category in sorted(ious.columns):
        column = ious[category]
        ap_at_iout=(column > args.iou_t).sum()/column.notnull().sum()
        print(f"{category: <10}\t{column.mean():0.2f}\t\t{column.median():0.2f}\t\t{ap_at_iout:0.2f}")

if __name__ == "__main__":
    main()
