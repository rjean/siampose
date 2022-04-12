# this module will export Objectron samples to HDF5 archives that can be randomly accessed
# (note: if a sequence contains two objects, only the first will be kept/exported)
# (note2: by default, only one of every X frames will be exported, with X=5 by default)

import h5py

from selfsupmotion.data.objectron.sequence_parser import *
from selfsupmotion.data.utils import *


if __name__ == "__main__":
    data_path = "/wdata/datasets/objectron/"

    # each object+sequence will be contained in its own group (under the name 'objtype/seqid')
    # all attributes will be encoded into separate datasets indexable by frame index
    # ... frame jumps might not be exactly 5-frame-apart (e.g. when blurred frames are skipped)

    # these fields will be encoded into datasets
    data_fields_to_export = [
        "IMAGE",
        "IMAGE_ID",
        "TIMESTAMP_MCSEC",
        "POINT_2D",
        "POINT_3D",
        "VIEW_MATRIX",
        # for some terrible reason the projection+intrinsic matrices change between frames???
        "PROJECTION_MATRIX",
        "INTRINSIC_MATRIX",
        "EXTRINSIC_MATRIX",
        "OBJECT_TRANSLATION",
        "OBJECT_ORIENTATION",
        "OBJECT_SCALE",
        "VISIBILITY",
        "PLANE_CENTER",
        "PLANE_NORMAL",
        # these two are non-standard:
        "BBOX_2D_IM",
        "CENTROID_2D_IM",
    ]

    # these fields will be encoded into group attributes
    attr_fields_to_export = [
        "IMAGE_WIDTH",
        "IMAGE_HEIGHT",
        "ORIENTATION",
    ]

    objectron_subset = "test"  # in ["train", "test"]
    target_subsampl_rate = 5  # jump and extract one out of every 5 frames
    hdf5_output_path = (
        data_path + f"extract_s{target_subsampl_rate}_raw_{objectron_subset}.hdf5.tmp"
    )

    with h5py.File(hdf5_output_path, "w") as fd:
        fd.attrs["target_subsampl_rate"] = target_subsampl_rate
        fd.attrs["objectron_subset"] = objectron_subset
        fd.attrs["orig_data_path"] = data_path
        fd.attrs["data_fields"] = data_fields_to_export
        fd.attrs["attr_fields"] = attr_fields_to_export

        for object in ObjectronSequenceParser.all_objects:
            parser = ObjectronSequenceParser(
                objectron_root=data_path, subset=objectron_subset, objects=[object]
            )
            try:

                for seq_idx, (seq_context, seq_data) in enumerate(parser):
                    seq_name = f"{object}/{seq_idx:05d}"
                    print(f"Processing: {seq_name}")
                    if objectron_subset == "train":
                        if object == "bike" and seq_idx in [182]:
                            continue  # bug in encoding (parsing blocks and goes oom)
                        if object == "book" and (
                            seq_idx in [24, 584, 1500] or seq_idx >= 1531
                        ):
                            continue  # bug in encoding (tfrecord readback goes boom)
                        if object == "bottle" and seq_idx in [666]:
                            continue  # bug in encoding (tfrecord readback goes boom)
                        if object == "chair" and seq_idx in [436, 1281, 1341]:
                            continue  # bug in encoding (tfrecord readback goes boom)
                    elif objectron_subset == "test":
                        # if object == "bike" and seq_idx in [182]:
                        #    continue  # bug in encoding (parsing blocks and goes oom)
                        pass

                    try:
                        sample_seq = ObjectronFrameParser(seq_context, seq_data)
                        frame_count = len(sample_seq)
                        max_dataset_len = (frame_count // target_subsampl_rate) + 1
                        assert max_dataset_len > 0
                        got_first_frame = False
                        last_frame_idx = None
                        curr_blur_threshold = 0.05
                        non_blurry_grad_vals = []
                        dataset_map = {}
                        captured_frame_idx = 0
                        for frame_idx, frame in enumerate(sample_seq):
                            if (
                                last_frame_idx is not None
                                and frame_idx - last_frame_idx < target_subsampl_rate
                            ):
                                continue

                            # check whether we want to keep this frame or not based on blurriness
                            if object in ObjectronSequenceParser.blurred_objects:
                                big_crop = get_obj_center_crop(frame, 0, (320, 320))
                                is_blurry, latest_val = is_frame_blurry(
                                    big_crop,
                                    nz_gradmag_threshold=curr_blur_threshold,
                                    return_nz_grad_mag=True,
                                )
                                if not is_blurry:
                                    non_blurry_grad_vals.append(latest_val)
                                    if len(non_blurry_grad_vals) > 10:
                                        non_blurry_grad_vals.pop(0)
                                    curr_blur_threshold = min(
                                        max(np.mean(non_blurry_grad_vals) * 0.6, 0.05),
                                        0.2,
                                    )
                                # cv.imshow("big crop", big_crop)
                                # cv.waitKey(1)
                                if is_blurry:
                                    continue

                            # collapse to a single instance if many are available (drops data, but rare)
                            if frame["INSTANCE_NUM"] > 1:
                                frame["POINT_NUM"] = frame["POINT_NUM"][0]
                                assert (
                                    frame["POINT_NUM"] == 9
                                )  # 9x 3D points per instance (should be const)
                                assert len(frame["POINT_2D"]) > 27
                                frame["POINT_2D"] = frame["POINT_2D"][:27]
                                assert len(frame["POINT_3D"]) > 27
                                frame["POINT_3D"] = frame["POINT_3D"][:27]
                                frame["OBJECT_TRANSLATION"] = frame[
                                    "OBJECT_TRANSLATION"
                                ][:3]
                                frame["OBJECT_ORIENTATION"] = frame[
                                    "OBJECT_ORIENTATION"
                                ][:9]
                                frame["OBJECT_SCALE"] = frame["OBJECT_SCALE"][:3]
                            frame["VISIBILITY"] = frame["VISIBILITY"][0]
                            frame["BBOX_2D_IM"] = frame["BBOX_2D_IM"][0]
                            frame["CENTROID_2D_IM"] = frame["CENTROID_2D_IM"][0]

                            # encode frame data, creating necessary datasets if needed
                            for field in data_fields_to_export:
                                data = frame[field]
                                if field == "IMAGE":
                                    retval, data = cv.imencode(
                                        ".jpg",
                                        data,
                                        params=(
                                            cv.IMWRITE_JPEG_OPTIMIZE,
                                            1,
                                            cv.IMWRITE_JPEG_QUALITY,
                                            90,
                                        ),
                                    )
                                    assert retval
                                    data = np.frombuffer(data, dtype=np.uint8)
                                    if not got_first_frame:
                                        dataset_map[field] = fd.create_dataset(
                                            name=seq_name + "/" + field,
                                            shape=(max_dataset_len,),
                                            maxshape=(max_dataset_len,),
                                            dtype=h5py.special_dtype(vlen=np.uint8),
                                        )
                                else:
                                    if not got_first_frame:
                                        if np.isscalar and not isinstance(
                                            data, np.ndarray
                                        ):
                                            data = np.asarray(data)
                                        dataset_map[field] = fd.create_dataset(
                                            name=seq_name + "/" + field,
                                            shape=(max_dataset_len, *data.shape),
                                            maxshape=(max_dataset_len, *data.shape),
                                            dtype=data.dtype,
                                        )
                                dataset_map[field][captured_frame_idx] = data

                            # encode/verify sequence attributes (should be const for all frames)
                            for field in attr_fields_to_export:
                                attr_val = frame[field]
                                if field == "ORIENTATION":  # decode into a str
                                    attr_val = attr_val[0].decode()
                                if not got_first_frame:
                                    fd[seq_name].attrs[field] = attr_val
                                else:
                                    assert np.array_equal(
                                        fd[seq_name].attrs[field], attr_val
                                    )

                            got_first_frame = True
                            last_frame_idx = frame_idx
                            captured_frame_idx += 1
                        assert captured_frame_idx > 0
                        for dataset in dataset_map.values():
                            dataset.resize(captured_frame_idx, axis=0)
                        fd.flush()
                    except Exception as e:
                        print(f"EXCEPTION CAUGHT FOR {seq_name}:\n{str(e)}")
                        continue
            except Exception as e:
                print(
                    f"EXCEPTION CAUGHT FOR {object} at seq_idx = {seq_idx}:\n{str(e)}"
                )
                continue
    print("all done")
