"""GSL: The Greek Sign Language Dataset"""

import csv
import os
from os import path


import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Dict
from tensorflow_datasets.core.download.checksums import UrlInfo
from tensorflow_datasets.core import utils

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig

_DESCRIPTION = """
The Greek Sign Language (GSL) is a large-scale RGB+D dataset, suitable for Sign Language Recognition (SLR) and Sign Language Translation (SLT). The video captures are conducted using an Intel RealSense D435 RGB+D camera at a rate of 30 fps. Both the RGB and the depth streams are acquired in the same spatial resolution of 848x480 pixels.

The dataset contains 10,290 sentence instances, 40,785 gloss instances, 310 unique glosses (vocabulary size) and 331 unique sentences, with 4.23 glosses per sentence on average. Each signer is asked to perform the pre-deﬁned dialogues ﬁve consecutive times.
"""

_CITATION = """
@misc{tsiami2022gsl,
  title = {{GSL: The Greek Sign Language Dataset}},
  author = {Tsiami, Ioanna and Efthimiou, Eleni and Dimou, Aimilios Dimitrios and Doukalopoulou, Dimitra and Goulas, Theodoros and Kouremenos, Dimitris},
  publisher = {IEEE Dataport},
  year = {2022},
  doi = {10.21227/wyk9-r981}
}
"""

_VIDEOS_URLS_TEMPLATE = "https://zenodo.org/records/4756317/files/{}.zip?download=1"
_DEPTH_URLS_TEMPLATE = "https://zenodo.org/records/4756317/files/{}_Depth.zip?download=1"


class GSL(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for GSL dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "2.0.0": "v2 public release",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=True),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {
            "id": tfds.features.Text(),
            "signer": tfds.features.Text(),
            "sentence": {
                "id": tfds.features.Text(),
                "text": tfds.features.Text(),
                "glosses": tfds.features.Sequence(tfds.features.Text()),
            },
            "instance": tf.int32,
            "video": self._builder_config.video_feature((848, 480)),
            "depth_video": self._builder_config.video_feature((848, 480), 1),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://vcl.iti.gr/dataset/gsl/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        supplementary_urls = [
            "https://zenodo.org/records/4756317/files/supplementary.zip?download=1",
            "https://zenodo.org/records/4756317/files/GSL_split.zip?download=1",
        ]

        supplementary_files = dl_manager.download_and_extract(supplementary_urls)
        split_dir = [file for file in supplementary_files if file.name == "GSL_split"][0]
        continuous_dir = os.path.join(split_dir, "GSL_continuous")

        video_names = [f"{prefix}{i}" for prefix in ["health", "kep", "police"] for i in range(1, 6)]
        depth_names = [f"{name}_Depth" for name in video_names]
        
        video_urls = [_VIDEOS_URLS_TEMPLATE.format(name) for name in video_names]
        depth_urls = [_DEPTH_URLS_TEMPLATE.format(name) for name in depth_names]

        download_urls = video_urls + depth_urls
        url_infos = self._get_url_infos(download_urls)
        downloaded_files = dl_manager.download_and_extract(url_infos)

        video_files = [file for file in downloaded_files if file.name in video_names]
        depth_files = [file for file in downloaded_files if file.name in depth_names]

        return {
            split: self._generate_examples(split, video_files, depth_files, os.path.join(continuous_dir, f"{split}.csv"))
            for split in ["GSL-SD-train", "GSL-SD-val", "GSL-SD-test"]
        }

    def _get_url_infos(self, urls: List[str]) -> Dict[str, UrlInfo]:
        """Returns UrlInfo objects for each URL."""
        with utils.try_with_retry(self._checksums_path.open, mode='r') as f:
            url_infos = {}
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) == 3:
                    url, checksum, _ = cols
                    if url in urls:
                        url_infos[url] = UrlInfo(
                            size=None,  # Set size to None since we don't have the file size
                            checksum=checksum,
                            filename=None
                        )
        return url_infos

    def _generate_examples(self, split, video_paths, depth_paths, split_path):
        """Yields examples."""
        with open(split_path, encoding="utf-8") as f:
            split_data = list(csv.DictReader(f))

        for row in split_data:
            video_id = row["video_id"]
            video_filename = f"{video_id}.mp4"
            depth_filename = f"{video_id}.mp4"

            video_path = [
                path
                for paths in video_paths
                for path in paths.iterdir()
                if path.name == video_filename
            ][0]
            depth_path = [
                path
                for paths in depth_paths
                for path in paths.iterdir()
                if path.name == depth_filename
            ][0]

            yield video_id, {
                "id": video_id,
                "signer": row["signer"],
                "sentence": {
                    "id": row["sentence"],
                    "text": row["translation"],
                    "glosses": row["annotation"].split(),
                },
                "instance": int(row["instance"]),
                "video": video_path,
                "depth_video": depth_path,
            }