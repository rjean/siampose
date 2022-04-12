import pickle
import shutil

import matplotlib.pyplot as plt
import sklearn
import umap

import selfsupmotion.data.objectron.dataset.graphics
import torch.utils.data
import torch.utils.tensorboard
import torchvision.models
import torchvision.transforms
import torchvision.transforms.functional

from selfsupmotion.data.objectron.sequence_parser import *
from selfsupmotion.data.objectron.utils import *
from selfsupmotion.data.utils import *


def extract_crop_data(
    data_path: typing.AnyStr,
    max_obj_count: int = 50,
    crop_count: int = 10,
) -> typing.Dict:
    output_data = {}
    for object in ["camera", "chair", "cup", "shoe"]:
        parser = ObjectronSequenceParser(data_path, objects=[object])
        for seq_idx, (seq_context, seq_data) in enumerate(parser):
            if seq_idx > max_obj_count:
                break
            sample_seq = ObjectronFrameParser(seq_context, seq_data)
            frame_count = len(sample_seq)
            frame_iter_offset = max(frame_count / crop_count, 1.0)
            next_frame_idx_to_extract = 0
            seq_name = f"{object}{seq_idx:05d}"
            output_data[seq_name] = {}
            for frame_idx, frame in enumerate(sample_seq):
                if frame_idx < next_frame_idx_to_extract:
                    continue
                next_frame_idx_to_extract += frame_iter_offset
                big_crop = get_obj_center_crop(frame, 0, (320, 320))
                rescaled_crop = cv.resize(big_crop, (224, 224))
                cv.imshow("crop", rescaled_crop)
                cv.waitKey(1)
                output_data[seq_name][frame_idx] = {**frame, "CROP": rescaled_crop}
    cv.destroyAllWindows()
    return output_data


def compute_resnet_embedding_without_fc(
    tensor: torch.Tensor,
    resnet: torch.nn.Module,
    apply_avg_pool: bool = True,
    apply_flatten: bool = True,
) -> torch.Tensor:
    x = resnet.conv1(tensor)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)
    if apply_avg_pool:
        x = resnet.avgpool(x)
        if apply_flatten:
            x = torch.flatten(x, 1)
    return x


def extract_embeddings(
    data_path: typing.AnyStr,
) -> typing.Dict:
    embeddings_backup_path = data_path + "embeddings.pkl"
    if not os.path.isfile(embeddings_backup_path):
        crops_backup_path = data_path + "crops.pkl"
        if not os.path.isfile(crops_backup_path):
            data = extract_crop_data(data_path)
            with open(crops_backup_path, "wb") as fd:
                pickle.dump(data, fd)
        else:
            with open(crops_backup_path, "rb") as fd:
                data = pickle.load(fd)
        resnet50_imnet = torchvision.models.resnet50(pretrained=True).eval()
        imnet_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        with torch.no_grad():
            for seq_name, seq_data in data.items():
                for frame_idx, frame_data in seq_data.items():
                    crop = cv.cvtColor(frame_data["CROP"], cv.COLOR_BGR2RGB)
                    tensor = torch.unsqueeze(imnet_transforms(crop), 0)
                    logits = torch.squeeze(resnet50_imnet(tensor), 0).numpy()
                    embed = compute_resnet_embedding_without_fc(tensor, resnet50_imnet)
                    embed = torch.squeeze(embed, 0).numpy()
                    frame_data["imnet_logits"], frame_data["imnet_embed"] = (
                        logits,
                        embed,
                    )
        with open(embeddings_backup_path, "wb") as fd:
            pickle.dump(data, fd)
    else:
        with open(embeddings_backup_path, "rb") as fd:
            data = pickle.load(fd)
    return data


def plot_clusters(
    data: typing.Dict,
    use_logits: bool = False,
    use_plt: bool = True,
    plot_pts_count: typing.Optional[int] = 40,
    use_umap: bool = False,
):
    points, meta, crops = [], [], []
    pts_key = "imnet_logits" if use_logits else "imnet_embed"
    for seq_name, seq_data in data.items():
        for frame_idx, frame_data in seq_data.items():
            if pts_key in frame_data:
                points.append(frame_data[pts_key])
                meta.append(seq_name)
                crop = cv.cvtColor(frame_data["CROP"], cv.COLOR_BGR2RGB)
                crops.append(torchvision.transforms.functional.to_tensor(crop))
    _, class_idxs = np.unique([m.split("0")[0] for m in meta], return_inverse=True)
    instance_names, instance_idxs = np.unique(meta, return_inverse=True)
    crops = torch.stack(crops)
    if use_umap:
        reducer = umap.UMAP(n_components=2 if use_plt else 3)
    else:
        reducer = sklearn.manifold.TSNE(n_components=2 if use_plt else 3)
    proj_pts = reducer.fit_transform(np.asarray(points))
    if plot_pts_count is not None:
        assert plot_pts_count <= len(instance_names)
        instance_idxs_subset = np.random.permutation(len(instance_names))[
            :plot_pts_count
        ]
        subset_mask = np.isin(instance_idxs, instance_idxs_subset)
        proj_pts = proj_pts[subset_mask]
        meta = np.asarray(meta)[subset_mask]
        crops = torch.from_numpy(crops.numpy()[subset_mask])
        class_idxs = class_idxs[subset_mask]
    if use_plt:
        plt.scatter(
            proj_pts[:, 0],
            proj_pts[:, 1],
            c=[get_label_color_mapping(x) / 255 for x in class_idxs],
        )
        plt.gca().set_aspect("equal", "datalim")
        plt.title(f"{'UMAP' if use_umap else 'TSNE'}-{pts_key}")
        plt.show()
    else:
        tb_dir_path = f"/tmp/runs/{pts_key}"
        if os.path.isdir(tb_dir_path):
            shutil.rmtree(tb_dir_path)
        os.makedirs(os.path.dirname(tb_dir_path), exist_ok=True)
        tb_writer = torch.utils.tensorboard.SummaryWriter(tb_dir_path)
        tb_writer.add_embedding(proj_pts, metadata=meta, label_img=crops)
        tb_writer.close()


if __name__ == "__main__":
    data_path = "/wdata/datasets/objectron/"
    data = extract_embeddings(data_path)
    plot_clusters(data)
    print("all done")
