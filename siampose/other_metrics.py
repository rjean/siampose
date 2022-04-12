import argparse
import numpy as np
import open3d as o3d
from open3d import JVisualizer
from PIL import Image
import pandas as pd
import plotly.express as px
from IPython import display
import siampose.zero_shot_pose as zsp
import siampose.geometry as geo
from sklearn.metrics import average_precision_score
import cupy as cp
from tqdm import tqdm
import cupy as cp
from sklearn.metrics import auc
from deco import synchronized, concurrent
from deco import synchronized, concurrent


seq_uid_count_map = {}
seq_uid_map = {}
frame_no_map = {}
experiment = None
# class OtherMetrics():
#    def __init__(self, experiment):
#        self.experiment = experiment
#        self.seq_uid_map = {}
#        for idx in range(len(experiment.info_df)):
#            seq_uid = experiment.info_df["sequence_uid"].iloc[idx]
#            self.seq_uid_map[idx]=seq_uid
#        self.frame_no_map = {}
#        for idx in range(len(experiment.info_df)):
#            frame_no = experiment.info_df["frame"].iloc[idx]
#            self.frame_no_map[idx]=frame_no
#        self.seq_uid_count_map = {}
#        for seq_uid in tqdm(experiment.info_df["sequence_uid"].unique()):
#            total_matches=(experiment.info_df["sequence_uid"]==seq_uid).sum()
#            self.seq_uid_count_map[seq_uid]=total_matches


def compute_all_results(experiment):
    global seq_uid_map, frame_no_map, seq_uid_count_map
    cuda_embeddings = cp.asarray(experiment.embeddings)
    embeddings = experiment.embeddings
    chunk_size = 500
    chunks = []
    results = []
    chunk_size = 100
    for i in tqdm(range(0, int(len(embeddings) / chunk_size) + 1)):
        start = i * chunk_size
        end = i * chunk_size + chunk_size
        if end > len(embeddings):
            end = len(embeddings)
        # start = chunk_size
        # end = chunk_size+chunk_size
        chunk_similarity = cp.dot(cuda_embeddings[start:end], cuda_embeddings.T)
        chunk_similarity[np.arange(chunk_size), np.arange(chunk_size) + start] = 0
        chunk_matches = cp.argsort(chunk_similarity, axis=1)
        chunk_matches = cp.flip(chunk_matches, axis=1)
        chunk_matches = chunk_matches.get()
        chunk_results = get_results_for_chunk(chunk_matches, start, end)
        results.append(chunk_results)
    compiled_results = {}
    for result in results:
        compiled_results.update(result)
    # compiled_results
    df = pd.DataFrame.from_dict(compiled_results).T
    df = df.rename(columns={0: "reid", 1: "jitter", 2: "AUC", 3: "seq_uid"})
    return df


@concurrent
def get_chunk_metrics(local_matches, the_idx, start, end):
    global seq_uid_map, frame_no_map, seq_uid_count_map  # , experiment
    # print(".")
    # all_matches  = np.argsort(similarity_matrix[the_idx],axis=0)[::-1]
    # seq_uid = experiment.info_df["sequence_uid"].iloc[the_idx]
    seq_uid = seq_uid_map[the_idx]
    # seq_uid, self.seq_uid_map[local_matches[0]], self.seq_uid_map[local_matches[0]], frame_no_map[the_idx], frame_no_map[local_matches[0]]
    frame_jitter = abs(frame_no_map[the_idx] - frame_no_map[local_matches[0]])
    reid = seq_uid == seq_uid_map[local_matches[0]]
    r_at_k = np.zeros(len(local_matches))
    p_at_k = np.zeros(len(local_matches))
    match_or_not = np.zeros(len(local_matches))
    total_matches = seq_uid_count_map[seq_uid]
    match_count = 0
    for k, match in enumerate(local_matches):
        # if seq_uid == experiment.info_df["sequence_uid"].iloc[match]:
        if seq_uid == seq_uid_map[match]:
            match_count += 1
            match_or_not[k] = 1
        else:
            match_or_not[k] = 0
        # r = match_count/total_matches
        # if last_r_at_k!=r:
        r_at_k[k] = match_count / total_matches
        p_at_k[k] = match_count / (k + 1)
    area_under_curve = auc(r_at_k, p_at_k)
    return (reid, frame_jitter, area_under_curve, seq_uid)


@synchronized
def get_results_for_chunk(chunk_matches, start, end):
    results = {}
    for the_idx in range(start, end):
        local_matches = chunk_matches[the_idx - start]
        results[the_idx] = get_chunk_metrics(local_matches, the_idx, start, end)
    return results


@concurrent
def get_metrics(cuda_embeddings, the_idx, seq_uid_map, frame_no_map, seq_uid_count_map):
    local_similarity = cp.dot(cuda_embeddings[the_idx], cuda_embeddings.T)
    # local_similarity = np.dot(embeddings[the_idx],embeddings.T)
    local_similarity[the_idx] = 0
    # local_matches  = np.argsort(similarity_matrix[the_idx],axis=0)[::-1]
    local_matches = np.argsort(local_similarity)[::-1]
    local_matches = local_matches.get()
    # all_matches  = np.argsort(similarity_matrix[the_idx],axis=0)[::-1]
    # seq_uid = experiment.info_df["sequence_uid"].iloc[the_idx]
    seq_uid = seq_uid_map[the_idx]
    seq_uid, seq_uid_map[local_matches[0]], frame_no_map[the_idx], frame_no_map[
        local_matches[0]
    ]
    frame_jitter = abs(frame_no_map[the_idx] - frame_no_map[local_matches[0]])
    reid = seq_uid == seq_uid_map[local_matches[0]]
    match_or_not = np.zeros(len(local_matches))
    r_at_k = np.zeros(len(local_matches))
    p_at_k = np.zeros(len(local_matches))
    total_matches = seq_uid_count_map[seq_uid]
    match_count = 0
    for k, match in enumerate(local_matches):
        # if seq_uid == experiment.info_df["sequence_uid"].iloc[match]:
        if seq_uid == seq_uid_map[match]:
            match_count += 1
            match_or_not[k] = 1
        else:
            match_or_not[k] = 0
        # r = match_count/total_matches
        # if last_r_at_k!=r:
        r_at_k[k] = match_count / total_matches
        p_at_k[k] = match_count / (k + 1)
    area_under_curve = auc(r_at_k, p_at_k)
    return (reid, frame_jitter, area_under_curve)


@synchronized
def process_data(experiment, seq_uid_map, frame_no_map, seq_uid_count_map):
    results = {}
    embeddings = experiment.embeddings
    cuda_embeddings = cp.asarray(embeddings)
    for the_idx in range(0, len(experiment.info_df[0:100])):
        results[the_idx] = get_metrics(
            cuda_embeddings, the_idx, seq_uid_map, frame_no_map, seq_uid_count_map
        )
    return results


import multiprocessing


def main():
    # multiprocessing.set_start_method('forkserver')
    global seq_uid_map, frame_no_map, seq_uid_count_map  # For multithreading
    # global args, experiment, ground_plane, symmetric, rescale, all_match_idxs, use_cupy
    parser = argparse.ArgumentParser(
        description="Command line tool for evaluating other metrics on objectron."
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment folder location. i.e. outputs/pretrain_224",
    )
    # parser.add_argument("--subset_size", type=int, default=1000, help="Number of samples for 3D IoU evaluation")
    # parser.add_argument("--ground_plane", default=True, help="If enabled, snap to ground plane")
    # parser.add_argument("--iou_t", default=0.5, help="IoU threshold required to consider a positive match")
    # parser.add_argument("--symmetric", action="store_true",help="Rotate symetric objects (cups, bottles) and keep maximum IoU.")
    # parser.add_argument("--rescale", action="store_true",help="Rescale 3d bounding box")
    # parser.add_argument("--random_bbox", action="store_true", help="Fit a randomly selected bbox instead of the nearest neighbor")
    # parser.add_argument("--random_bbox_same", action="store_true", help="Fit a randomly selected bbox from same category instead of the nearest neighbor")
    # parser.add_argument("--trainset-ratio", type=float, default=1, help="Ratio of the training set sequences used for inference")
    # parser.add_argument("--single_thread", action="store_true", help="Disable multithreading.")
    # parser.add_argument("--cpu", action="store_true", help="Disable cuda accelaration.")
    # parser.add_argument("--no_align_axis", action="store_true", help="Don't to to align axis with ground plane.")
    # parser.add_argument("--legacy", action="store_true", help="Deprecated legacy evalution mode")
    parser.add_argument(
        "--test", action="store_true", help="Evaluate on test set embeddings"
    )
    parser.add_argument(
        "--class-accuracy", action="store_true", help="Evaluate on test set embeddings"
    )
    parser.add_argument(
        "--sequence", action="store_true", help="Sequence re-id metrics"
    )
    args = parser.parse_args()
    # TODO: Refactor the experiment handler outside of the Zero Shot Pose module.
    print(f"Running for experiment {args.experiment}")
    experiment = zsp.ExperimentHandlerFile(args.experiment, test=args.test)

    if args.class_accuracy:
        experiment.info_df["category"] = experiment.info_df["sequence_uid"].str.extract(
            "(.*?)/"
        )
        experiment.train_info_df["category"] = experiment.train_info_df[
            "sequence_uid"
        ].str.extract("(.*?)/")
        matches = zsp.find_all_match_idx(
            experiment.embeddings, experiment.train_embeddings
        )

        res = [
            experiment.train_info_df["category"].iloc[matches[i]]
            for i in list(experiment.info_df.index)
        ]
        experiment.info_df["result_category"] = res
        num_matches = (
            experiment.info_df["result_category"] == experiment.info_df["category"]
        ).sum()
        classification_accuracy = 100 * num_matches / len(experiment.info_df)

        print(f"Classification accuracy: {classification_accuracy:0.2f}%")

    # Fix parsing. TODO: Migrate to zsp
    experiment.info_df["category"] = experiment.info_df["sequence_uid"].str.extract(
        "(.*?)/"
    )
    experiment.train_info_df["category"] = experiment.train_info_df[
        "sequence_uid"
    ].str.extract("(.*?)/")
    experiment.info_df["frame"] = (
        experiment.info_df["uid"].str.extract("hdf5_\w+/\d+_(\d+)").astype(int)
    )

    seq_uid_map = {}
    for idx in range(len(experiment.info_df)):
        seq_uid = experiment.info_df["sequence_uid"].iloc[idx]
        seq_uid_map[idx] = seq_uid
    frame_no_map = {}
    for idx in range(len(experiment.info_df)):
        frame_no = experiment.info_df["frame"].iloc[idx]
        frame_no_map[idx] = frame_no
    seq_uid_count_map = {}
    for seq_uid in tqdm(experiment.info_df["sequence_uid"].unique()):
        total_matches = (experiment.info_df["sequence_uid"] == seq_uid).sum()
        seq_uid_count_map[seq_uid] = total_matches

    if args.sequence:
        # othermetrics = OtherMetrics(experiment)
        df = compute_all_results(experiment)
        print(f"Re-id: {df['reid'].mean()*100:0.2f}")
        print(f"Jitter: {df['jitter'].mean():0.2f}")
        print(f"AUC: {df['AUC'].mean():0.2f}")
        df.to_csv(f"{args.experiment}/othermetrics.csv")
    # df["jitter"].mean()
    # df["AUC"].mean()

    # results = process_data(experiment, seq_uid_map, frame_no_map, seq_uid_count_map)

    # Same sequence.
    # match_uid = [experiment.info_df["uid"].iloc[matches[i]] for i in list(experiment.info_df.index)]
    # experiment.info_df["match_uid"] = match_uid
    # experiment.info_df["match_sequence_uid"] = experiment.info_df["match_uid"].str.extract("hdf5_(\w+/\d+)_")
    # experiment.info_df["match_frame"] = experiment.info_df["match_uid"].str.extract("hdf5_\w+/\d+_(\d+)").astype(int)
    # experiment.info_df["frame"] = experiment.info_df["uid"].str.extract("hdf5_\w+/\d+_(\d+)").astype(int)
    # embeddings=experiment.embeddings
    # cuda_embeddings = cp.asarray(embeddings)

    # same_sequence = (experiment.info_df["match_sequence_uid"]==experiment.info_df["sequence_uid"])
    # sequence_reid = 100*same_sequence.sum()/len(experiment.info_df)
    # print(f"Sequence re-identification accuracy: {sequence_reid:0.2f}%")


if __name__ == "__main__":
    main()
