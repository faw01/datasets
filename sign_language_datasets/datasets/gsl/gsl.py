"""Greek Sign Language (GSL) Dataset"""

import os
import tensorflow_datasets as tfds
from tensorflow.io.gfile import GFile

from sign_language_datasets.datasets.warning import dataset_warning

_DESCRIPTION = """
The Greek Sign Language (GSL) is a large-scale RGB+D dataset, suitable for Sign Language Recognition (SLR) and Sign Language Translation (SLT).
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

class GSL(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for GSL dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "2.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "gloss": tfds.features.Text(),
                "video_path": tfds.features.Text(),
                "depth_path": tfds.features.Text(),
            }),
            homepage="https://vcl.iti.gr/dataset/gsl/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        gsl_dir = os.path.join(data_dir, "gsl")
        split_dir = os.path.join(data_dir, "GSL_split", "GSL_continuous")

        zip_files = [f for f in os.listdir(gsl_dir) if f.endswith('.zip')]
        
        file_dict = {f: os.path.join(gsl_dir, f) for f in zip_files}
        
        downloaded_files = dl_manager.download(file_dict)

        return {
            "train": self._generate_examples(os.path.join(split_dir, "GSL-SD-train.txt"), downloaded_files),
            "validation": self._generate_examples(os.path.join(split_dir, "GSL-SD-val.txt"), downloaded_files),
            "test": self._generate_examples(os.path.join(split_dir, "GSL-SD-test.txt"), downloaded_files),
        }

    def _generate_examples(self, split_file: str, downloaded_files: dict):
        """Yields examples."""
        with GFile(split_file, "r") as f:
            for i, line in enumerate(f):
                video_id, gloss = line.strip().split("|")
                scenario = video_id.split("_")[0]

                video_file = f"{scenario}.zip"
                depth_file = f"{scenario}_Depth.zip"

                yield i, {
                    "id": video_id,
                    "gloss": gloss,
                    "video_path": str(downloaded_files[video_file]),
                    "depth_path": str(downloaded_files[depth_file]),
                }