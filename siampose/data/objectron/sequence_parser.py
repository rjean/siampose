"""Objectron dataset parser module.

This module contains dataset parsers used to load the TFRecords of the Objectron dataset.
See https://github.com/google-research-datasets/Objectron for more info.
"""

import os
import typing

import cv2 as cv
import numpy as np

import siampose.data.objectron.schema.features
import tensorflow as tf
import tensorboard as tb
import torch.utils.data
import torch.utils.tensorboard

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class ObjectronSequenceParser(torch.utils.data.Dataset):
    """Objectron sequence dataset parser.

    This class can be used to parse the Objectron TFRecord Sequences. It cannot work with the
    shuffled version of the dataset, or with the raw videos. It relies on the Objectron utility
    modules for parsing. It returns the raw data of each sequence individually for more processing.

    If sequences are missing (e.g. if the dataset was only partially downloaded), the parser will
    try to return only valid sequences.
    """

    all_objects = [
        "bike", "book", "bottle", "camera", "cereal_box", "chair", "cup", "laptop", "shoe",
    ]

    blurred_objects = ["bike"]  # see https://github.com/google-research-datasets/Objectron/issues/10

    def __init__(
            self,
            objectron_root: typing.AnyStr,
            subset: typing.AnyStr = "train",
            objects: typing.Optional[typing.Sequence[typing.AnyStr]] = None,  # default => use all
            compute_tot_len: bool = False,
    ):
        assert subset in ["train", "test"], "invalid objectron subset"
        self.root_path = objectron_root
        assert os.path.exists(self.root_path), f"invalid root dataset path: {self.root_path}"
        self.sequences_path = os.path.join(self.root_path, "sequences")
        assert os.path.exists(self.sequences_path), f"invalid sequences path: {self.sequences_path}"
        if not objects:
            self.objects = self.all_objects
        else:
            assert all([obj in self.all_objects for obj in objects]), "invalid object name used in filter"
            self.objects = objects
        self.shards_map = {
            c: sorted(tf.io.gfile.glob(self.sequences_path + f"/{c}/{c}_{subset}*"))
            for c in self.objects
        }
        self.dataset = tf.data.TFRecordDataset([s for shards in self.shards_map.values() for s in shards])
        self.sequence_count = None
        if compute_tot_len:
            self.sequence_count = self.dataset.reduce(np.int64(0), lambda x, _: x + 1)
        self.dataset = self.dataset.map(self._parse_sequence)

    def _parse_sequence(self, record):
        context, data = tf.io.parse_single_sequence_example(
            serialized=record,
            sequence_features=selfsupmotion.data.objectron.schema.features.SEQUENCE_FEATURE_MAP,
            context_features=selfsupmotion.data.objectron.schema.features.SEQUENCE_CONTEXT_MAP
        )
        context["objectron_root"] = self.root_path
        return context, data

    def __len__(self):
        if self.sequence_count is not None:
            return self.sequence_count
        raise NotImplementedError

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, item):
        raise NotImplementedError  # TFRecords are bad for indexing


class ObjectronFrameParser(torch.utils.data.Dataset):
    """Objectron frame dataset parser. Used to iterate over frames of TFRecord sequences.

    This object is intended to be lightweight enough to be create on-the-spot as TFRecords are
    read and converted into sequence tuples. Each tuple (passed into the constructor) is
    composed of a context dictionary (with high-level variables) and a data dictionary (that
    contains the sequence data itself).

    The frame data will be returned as a dictionary of values. The PNG images will be decoded
    using OpenCV, and returned as a numpy array inside that dictionary. All other tensors will
    also be converted to numpy arrays for non-TensorFlow interoperability.

    See the `objectron.schema.features` module for the exact list of features produced per frame.
    """

    def __init__(
            self,
            context: typing.Dict,
            data: typing.Dict,
    ):
        assert isinstance(context, dict) and isinstance(data, dict)
        assert all([s in context for s in ["count", "sequence_id", "objectron_root"]])
        assert all([s in data for s in ["image/encoded", "point_2d", "point_3d"]])  # there's more
        self.root_path = context["objectron_root"]
        self.sequence_id = context["sequence_id"].numpy().decode()
        self.frame_count = int(context["count"])
        assert self.frame_count > 0
        self.data = data

    def __len__(self):
        return self.frame_count

    def __iter__(self):
        for idx in range(self.frame_count):
            yield self[idx]

    def __getitem__(self, idx):
        assert 0 <= idx < self.frame_count
        # here, we will simply handle the different tensor types and convert them, but nothing more
        out_sample = {}
        for feat_name, feat_code in selfsupmotion.data.objectron.schema.features.FEATURE_NAMES.items():
            assert feat_code in selfsupmotion.data.objectron.schema.features.FEATURE_MAP
            if feat_code not in self.data:
                continue
            feat_type = selfsupmotion.data.objectron.schema.features.FEATURE_MAP[feat_code]
            assert isinstance(feat_type, (tf.io.FixedLenFeature, tf.io.VarLenFeature))
            if isinstance(feat_type, tf.io.FixedLenFeature):
                assert feat_type.dtype in [tf.string, tf.int64]
                if feat_type.dtype == tf.string:
                    assert len(feat_type.shape) == 0
                    if feat_name == "IMAGE_ENCODED":
                        feat_name = "IMAGE"  # overwrite feat name, it won't be encoded anymore
                        buf = np.frombuffer(self.data[feat_code][idx].numpy(), dtype=np.uint8)
                        feat_val = cv.imdecode(buf, cv.IMREAD_UNCHANGED)
                    else:
                        feat_val = self.data[feat_code][idx].numpy().decode()
                else:  # if feat_type.dtype == tf.int64:
                    if feat_name == "TIMESTAMP_MCSEC":  # manual handling for bugged spec
                        feat_val = int(self.data[feat_code][idx])
                    elif len(feat_type.shape) == 0:
                        feat_val = int(self.data[feat_code])
                    else:
                        assert len(feat_type.shape) == 1 and feat_type.shape[0] == 1
                        feat_val = int(self.data[feat_code][idx])
            else:  # isinstance(feat_type, tf.io.VarLenFeature):
                assert feat_type.dtype in [tf.string, tf.int64, tf.float32]
                tensor = self.data[feat_code]
                if isinstance(tensor, tf.SparseTensor):
                    tensor = tensor.values.numpy().reshape(tensor.shape)
                assert tensor.shape[0] == self.frame_count
                feat_val = tensor[idx]
            if isinstance(feat_val, np.ndarray) and np.isscalar(feat_val):
                feat_val = feat_val.item()
            out_sample[feat_name] = feat_val
        # we will add an extra feature here: the (2d) bounding box & centroids of the objects
        # note: BBOX = [(min_X, min_Y), (max_X, max_Y)]  (in image coordinates)
        assert len(out_sample["POINT_NUM"]) == out_sample["INSTANCE_NUM"]
        out_sample["BBOX_2D_IM"] = [None] * out_sample["INSTANCE_NUM"]
        out_sample["CENTROID_2D_IM"] = [None] * out_sample["INSTANCE_NUM"]
        point_idx_offset = 0
        frame_pts = out_sample["POINT_2D"].reshape((-1, 3))
        im_scale = np.asarray((out_sample["IMAGE"].shape[1], out_sample["IMAGE"].shape[0]), np.float32)
        for instance_idx, instance_point_count in enumerate(out_sample["POINT_NUM"]):
            instance_pts = frame_pts[point_idx_offset:point_idx_offset + instance_point_count]
            bbox_min = tuple(np.multiply([instance_pts[:, i].min() for i in range(2)], im_scale))
            bbox_max = tuple(np.multiply([instance_pts[:, i].max() for i in range(2)], im_scale))
            centroid = tuple(np.multiply([instance_pts[:, i].mean() for i in range(2)], im_scale))
            out_sample["BBOX_2D_IM"][instance_idx] = (bbox_min, bbox_max)
            out_sample["CENTROID_2D_IM"][instance_idx] = centroid
            point_idx_offset += instance_point_count
        out_sample["BBOX_2D_IM"] = np.asarray(out_sample["BBOX_2D_IM"])
        out_sample["CENTROID_2D_IM"] = np.asarray(out_sample["CENTROID_2D_IM"])
        return out_sample
