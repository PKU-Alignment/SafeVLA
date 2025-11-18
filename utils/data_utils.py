import gzip
import json
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import h5py
from tqdm import tqdm

from utils.string_utils import (
    convert_byte_to_string,
    get_natural_language_spec,
    json_templated_spec_to_dict,
)


def read_jsonlgz(path: str, max_lines: Optional[int] = None) -> List[bytes]:
    with gzip.open(path, "r") as f:
        lines = []
        for line in tqdm(f, desc=f"Loading {path}"):
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


# create JsonType
JsonType = Union[str, bytes]


class LazyJsonDataset:
    """Lazily load a list of json data."""

    def __init__(self, data: List[JsonType]) -> None:
        """
        Inputs:
            data: a list of json documents
        """
        self.data = data
        self.cached_data: Dict[int, Union[List, Dict]] = {}

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        if index not in self.cached_data:
            self.cached_data[index] = json.loads(self.data[index])
        return self.cached_data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return "LazyJsonDataset(num_samples={}, cached_samples={})".format(
            len(self), len(self.cached_data)
        )

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            if i not in self.cached_data:
                self.cached_data[i] = json.loads(x)
            yield self.cached_data[i]

    def select(self, indices: Sequence[int]) -> "LazyJsonDataset":
        """Return a new dataset containing only the given indices."""
        return LazyJsonDataset(
            data=[self.data[i] for i in indices],
        )


class LazyJsonHouses(LazyJsonDataset):
    """Lazily load the a list of json houses."""

    def __init__(self, data: List[JsonType]) -> None:
        super().__init__(data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return "LazyJsonHouses(num_houses={}, cached_houses={})".format(
            len(self), len(self.cached_data)
        )

    def select(self, indices: Sequence[int]) -> "LazyJsonHouses":
        """Return a new dataset containing only the given indices."""
        return LazyJsonHouses(
            data=[self.data[i] for i in indices],
        )

    @staticmethod
    def from_jsonlgz(path: str, max_houses: Optional[int] = None) -> "LazyJsonHouses":
        """Load the houses from a .jsonl.gz file."""
        return LazyJsonHouses(data=read_jsonlgz(path=path, max_lines=max_houses))

    @staticmethod
    def from_dir(
        house_dir: str,
        subset: Literal["train", "val", "test"],
        max_houses: Optional[int] = None,
    ) -> "LazyJsonHouses":
        """Load the houses from a directory containing {subset}.jsonl.gz files."""
        return LazyJsonHouses.from_jsonlgz(
            path=os.path.join(house_dir, f"{subset}.jsonl.gz"),
            max_houses=max_houses,
        )


class LazyJsonTaskSpecs(LazyJsonDataset):
    """Lazily load a list of json task specs."""

    def __init__(self, data: List[JsonType]) -> None:
        super().__init__(data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return "LazyJsonTaskSpecs(num_tasks={}, cached_tasks={})".format(
            len(self), len(self.cached_data)
        )

    def select(self, indices: Sequence[int]) -> "LazyJsonHouses":
        """Return a new dataset containing only the given indices."""
        return LazyJsonTaskSpecs(
            data=[self.data[i] for i in indices],
        )

    @staticmethod
    def from_jsonlgz(path: str, max_task_specs: Optional[int] = None) -> "LazyJsonTaskSpecs":
        """Load the tasks from a .jsonl.gz file."""
        return LazyJsonTaskSpecs(data=read_jsonlgz(path=path, max_lines=max_task_specs))

    @staticmethod
    def from_dir(
        task_spec_dir: str,
        subset: Literal["train", "val", "test"],
        max_task_specs: Optional[int] = None,
    ) -> "LazyJsonTaskSpecs":
        """Load the task specs from a directory containing {subset}.jsonl.gz files."""
        return LazyJsonTaskSpecs.from_jsonlgz(
            path=os.path.join(task_spec_dir, f"{subset}.jsonl.gz"),
            max_task_specs=max_task_specs,
        )


class DatasetDict(dict):
    """A dictionary-like object for storing datasets with attribute access."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def load_dataset_from_path(
    path_to_splits: Optional[str] = None,
    split_to_path: Optional[Dict[str, str]] = None,
    max_items_per_split: Optional[Union[int, Dict[str, int]]] = None,
) -> DatasetDict:
    """从本地路径加载数据集（不依赖prior库）
    
    Args:
        path_to_splits: 包含train/val/test子目录的路径
        split_to_path: split名称到路径的映射字典
        max_items_per_split: 每个split最多加载的项目数
        
    Returns:
        DatasetDict: 包含各个split的LazyJsonDataset对象的字典
        
    Examples:
        # 从目录加载
        data = load_dataset_from_path(path_to_splits="/path/to/data")
        
        # 从指定文件加载
        data = load_dataset_from_path(
            split_to_path={
                "train": "/path/to/train.jsonl.gz",
                "val": "/path/to/val.jsonl.gz"
            }
        )
    """
    assert (path_to_splits is None) != (split_to_path is None), \
        "必须且只能提供 path_to_splits 或 split_to_path 中的一个"
    
    # 处理 max_items_per_split
    if not isinstance(max_items_per_split, dict):
        max_items_per_split = defaultdict(lambda: max_items_per_split)
    else:
        max_items_per_split = defaultdict(lambda: None, max_items_per_split)
    
    # 构建 split_to_path
    if path_to_splits is not None:
        if not os.path.exists(path_to_splits):
            raise FileNotFoundError(f"路径不存在: {path_to_splits}")
        
        split_to_path = {
            "train": os.path.join(path_to_splits, "train"),
            "val": os.path.join(path_to_splits, "val"),
            "test": os.path.join(path_to_splits, "test"),
        }
    
    # 检查split路径是否存在
    valid_splits = {}
    for split, path in split_to_path.items():
        if os.path.exists(path):
            valid_splits[split] = path
        else:
            warnings.warn(f"Split '{split}' 路径不存在: {path}，将被跳过")
    
    if len(valid_splits) == 0:
        raise ValueError("没有找到任何有效的split")
    
    # 加载各个split的数据
    split_to_dataset = {}
    for split, path in valid_splits.items():
        max_lines = max_items_per_split[split]
        
        if path.endswith('.jsonl.gz'):
            # 直接是 .jsonl.gz 文件
            print(f"加载 {split} split: {path}")
            data = read_jsonlgz(path=path, max_lines=max_lines)
            split_to_dataset[split] = LazyJsonDataset(data=data)
        elif os.path.isdir(path):
            # 是目录，查找 .jsonl.gz 文件
            jsonlgz_files = [f for f in os.listdir(path) if f.endswith('.jsonl.gz')]
            if jsonlgz_files:
                # 假设目录下只有一个 .jsonl.gz 文件，或取第一个
                file_path = os.path.join(path, jsonlgz_files[0])
                print(f"加载 {split} split: {file_path}")
                data = read_jsonlgz(path=file_path, max_lines=max_lines)
                split_to_dataset[split] = LazyJsonDataset(data=data)
            else:
                warnings.warn(f"目录 {path} 中没有找到 .jsonl.gz 文件")
        else:
            warnings.warn(f"不支持的路径类型: {path}")
    
    return DatasetDict(**split_to_dataset)


def load_hdf5_sensor(path):
    if not os.path.isfile(path):
        return []

    data = []
    with h5py.File(path, "r") as d:
        for k in d.keys():
            j = json_templated_spec_to_dict(
                convert_byte_to_string(d[k]["templated_task_spec"][0, :])
            )
            j["house_index"] = int(d[k]["house_index"][0])
            last_agent_location = d[k]["last_agent_location"][0]
            j["agent_starting_position"] = [
                last_agent_location[0],
                last_agent_location[1],
                last_agent_location[2],
            ]
            j["agent_y_rotation"] = last_agent_location[4]
            j["natural_language_spec"] = get_natural_language_spec(j["task_type"], j)
            data.append(j)
    return data


class Hdf5TaskSpecs:
    """Load hdf5_sensors.hdf5 data stored as {dataset_dir}/*/hdf5_sensors.hdf5."""

    def __init__(
        self,
        subset_dir: str,
        data: Optional[List[Dict]] = None,
        proc_id: Optional[int] = None,
        total_procs: Optional[int] = None,
        max_house_id: Optional[int] = None,
        max_task_specs: Optional[int] = None,
    ) -> None:
        """
        Inputs:
            subset_dir: path to the directory containing subdirectories with hdf5_sensors.hdf5 files
        """
        self.subset_dir = subset_dir
        self.proc_id = proc_id if proc_id is not None else 0
        self.total_procs = total_procs if total_procs is not None else 1
        self.max_house_id = max_house_id

        if data is None:
            # subdirs are zfilled house ids
            subdirs = sorted(os.listdir(self.subset_dir))
            if self.max_house_id is not None:
                subdirs = [subdir for subdir in subdirs if int(subdir) < self.max_house_id]

            # select paths for the current process
            paths = [
                f"{self.subset_dir}/{subdir}/hdf5_sensors.hdf5"
                for i, subdir in enumerate(subdirs)
                if i % self.total_procs == self.proc_id
            ]
            self.data = self.read_hdf5_sensors(paths)
        else:
            self.data = data

        self.max_task_specs = max_task_specs if max_task_specs is not None else len(self.data)
        self.data = self.data[: self.max_task_specs]

    def read_hdf5_sensors(self, paths) -> List[Dict]:
        data = []
        desc = (
            f"[proc: {self.proc_id}/{self.total_procs}] "
            f"Loading hdf5_sensors.hdf5 files from {self.subset_dir}"
        )
        for path in tqdm(paths, desc=desc):
            data.extend(load_hdf5_sensor(path))

        return data

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        return self.data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return "Hdf5TaskSpecs(num_samples={},proc_id={},total_procs={})".format(
            len(self), self.proc_id, self.total_procs
        )

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            yield x

    def select(self, indices: Sequence[int]) -> "Hdf5TaskSpecs":
        """Return a new dataset containing only the given indices."""
        return Hdf5TaskSpecs(
            subset_dir=self.subset_dir,
            data=[self.data[i] for i in indices],
            proc_id=self.proc_id,
            total_procs=self.total_procs,
        )

    def from_dataset_dir(
        dataset_dir: str,
        subset: Literal["train", "val", "test"],
        proc_id: Optional[int] = None,
        total_procs: Optional[int] = None,
        max_house_id: Optional[int] = None,
        max_task_specs: Optional[int] = None,
    ) -> "Hdf5TaskSpecs":
        """Load the tasks from a directory containing {dataset_dir}/{subset}/*/hdf5_sensors.hdf5 files."""
        return Hdf5TaskSpecs(
            subset_dir=os.path.join(dataset_dir, subset),
            proc_id=proc_id,
            total_procs=total_procs,
            max_house_id=max_house_id,
            max_task_specs=max_task_specs,
        )


if __name__ == "__main__":
    from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR

    houses = LazyJsonHouses.from_dir(
        OBJAVERSE_HOUSES_DIR,
        subset="train",
        max_houses=10,
    )
    print(houses)
    # task_specs = LazyJsonTaskSpecs.from_dir(
    #     "/root/data/ObjectNavType_Poliformer",
    #     "train",
    #     max_task_specs=10,
    # )
    task_specs = Hdf5TaskSpecs.from_dataset_dir(
        "/root/vida_datasets/pointing_data/GoNearPoint", "train", proc_id=2, total_procs=48
    )

    print(task_specs)

    import ipdb

    ipdb.set_trace()
