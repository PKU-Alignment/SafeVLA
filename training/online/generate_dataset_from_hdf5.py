import jsonlines, gzip, os, h5py
from tqdm import tqdm
from utils.string_utils import (
    get_natural_language_spec,
    json_templated_spec_to_dict,
    convert_byte_to_string,
)


def generate_RL_dataset(path, jsonlines_gz_data):
    all_data = ["{}/{}/hdf5_sensors.hdf5".format(path, data) for data in sorted(os.listdir(path))]
    datasets = []
    for i in tqdm(range(len(all_data))):
        data = all_data[i]
        if not os.path.isfile(data):
            continue
        d = h5py.File(data)
        for k in d.keys():
            j = json_templated_spec_to_dict(
                convert_byte_to_string(d[k]["templated_task_spec"][0, :])
            )
            j["house_index"] = int(d[k]["house_index"][0])
            j["agent_starting_position"] = [
                d[k]["last_agent_location"][0][0],
                d[k]["last_agent_location"][0][1],
                d[k]["last_agent_location"][0][2],
            ]
            j["agent_y_rotation"] = d[k]["last_agent_location"][0][4]
            j["natural_language_spec"] = get_natural_language_spec(j["task_type"], j)
            datasets.append(j)

    with gzip.open(jsonlines_gz_data, "wb") as f:
        json_writer = jsonlines.Writer(f)
        json_writer.write_all(datasets)


if __name__ == "__main__":
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/procthor_objectnav/ObjectNavType/train",
    #     "/net/nfs/prior/datasets/vida_datasets/procthor_objectnav/ObjectNavType/procthor_objectnav_train.jsonl.gz",
    # )
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/procthor_objectnav/ObjectNavType/val",
    #     "/net/nfs/prior/datasets/vida_datasets/procthor_objectnav/ObjectNavType/procthor_objectnav_val.jsonl.gz",
    # )
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/khzeng_boxnav_forRL/ObjectNavMulti/train",
    #     "/net/nfs/prior/datasets/vida_datasets/khzeng_boxnav_forRL/ObjectNavMulti/train.jsonl.gz",
    # )

    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/kianae_astar_goals_spoc_02MAY2024/EasyObjectNavType/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/EasyObjectNavType/val.jsonl.gz",
    # )

    target_folder = "/net/nfs.cirrascale/prior/datasets/vida_datasets/jiaheng_fetch_17July2024/"
    target_tasks = ['ObjectNavLocalRef', 'ObjectNavRelAttribute', 'RoomNav', 'ObjectNavRoom',
                    'ObjectNavDescription', 'ObjectNavAffordance']
    save_folder = "/net/nfs/prior/datasets/vida_datasets/jhu_rl/"

    for target_task in target_tasks:
        generate_RL_dataset(
            target_folder + target_task + "/train",
            save_folder + target_task + "/train.jsonl.gz",
        )

        generate_RL_dataset(
            target_folder + target_task + "/val",
            save_folder + target_task + "/val.jsonl.gz",
        )

    # # Locl Ref
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavLocalRef/train",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavLocalRef/train.jsonl.gz",
    # )
    #
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavLocalRef/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavLocalRef/val.jsonl.gz",
    # )
    #
    # # Rel Attribute
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavRelAttribute/train",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavRelAttribute/train.jsonl.gz",
    # )
    #
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavRelAttribute/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavRelAttribute/val.jsonl.gz",
    # )

    # # RoomNav
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/RoomNav/train",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/RoomNav/train.jsonl.gz",
    # )
    #
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/RoomNav/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/RoomNav/val.jsonl.gz",
    # )
    #
    # # ObjectNavRoom
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavRoom/train",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavRoom/train.jsonl.gz",
    # )
    #
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavRoom/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavRoom/val.jsonl.gz",
    # )
    #
    # # ObjectNavDescription
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavDescription/train",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavDescription/train.jsonl.gz",
    # )
    #
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavDescription/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavDescription/val.jsonl.gz",
    # )
    #
    # # ObjectNavAffordance
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavAffordance/train",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavAffordance/train.jsonl.gz",
    # )
    #
    # generate_RL_dataset(
    #     "/net/nfs/prior/datasets/vida_datasets/vida_clean_dataset/fifteen/ObjectNavAffordance/val",
    #     "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavAffordance/val.jsonl.gz",
    # )
