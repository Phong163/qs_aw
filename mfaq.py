# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import json
import os

import datasets


_CITATION = """\
@InProceedings{mfaq_a_multilingual_dataset,
    title={MFAQ: a Multilingual FAQ Dataset},
    author={Maxime {De Bruyn} and Ehsan Lotfi and Jeska Buhmann and Walter Daelemans},
    year={2021},
    booktitle={MRQA @ EMNLP 2021}
}
"""


_DESCRIPTION = """\
We present the first multilingual FAQ dataset publicly available. We collected around 6M FAQ pairs from the web, in 21 different languages.
"""

_HOMEPAGE = ""

_LICENSE = ""


_LANGUAGES = ["cs", "da", "de", "en", "es", "fi", "fr", "he", "hr", "hu", "id", "it", "nl", "no", "pl", "pt", "ro", "ru", "sv", "tr", "vi"]
_URLs = {}
_URLs.update({f"{l}": {"train": [f"data/{l}/train.jsonl"], "valid": [f"data/{l}/valid.jsonl"]} for l in _LANGUAGES})
_URLs["all"] = {"train": [f"data/{l}/train.jsonl" for l in _LANGUAGES], "valid": [f"data/{l}/valid.jsonl" for l in _LANGUAGES]}
_URLs.update({f"{l}_flat": {"train": [f"data/{l}/train.jsonl"], "valid": [f"data/{l}/valid.jsonl"]} for l in _LANGUAGES})
_URLs["all_flat"] = {"train": [f"data/{l}/train.jsonl" for l in _LANGUAGES], "valid": [f"data/{l}/valid.jsonl" for l in _LANGUAGES]}


class MFAQ(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = list(map(lambda x: datasets.BuilderConfig(name=x, version=datasets.Version("1.1.0")), _URLs.keys()))
    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        if "_flat" in self.config.name: 
            features = datasets.Features(
                {
                    "domain_id": datasets.Value("int64"),
                    "pair_id": datasets.Value("int64"),
                    "language": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string")
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "language": datasets.Value("string"),
                    "num_pairs": datasets.Value("int64"),
                    "domain": datasets.Value("string"),
                    "qa_pairs": [
                        {
                            "question": datasets.Value("string"), 
                            "answer": datasets.Value("string"),
                            "language": datasets.Value("string")
                        }
                    ]
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": data_dir["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": data_dir["valid"], "split": "valid"},
            ),
        ]

    def _generate_examples(
        self, filepaths, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for _id, row in enumerate(f):
                    data = json.loads(row)
                    if "flat" in self.config.name:
                        for i, pair in enumerate(data["qa_pairs"]):
                            yield f"{filepath}_{_id}_{i}", {
                                "domain_id": data["id"],
                                "pair_id": i,
                                "domain": data["domain"],
                                "language": data["language"],
                                "question": pair["question"],
                                "answer": pair["answer"]
                            }
                    else:
                        yield f"{filepath}_{_id}", {
                            "id": data["id"],
                            "domain": data["domain"],
                            "language": data["language"],
                            "num_pairs": data["num_pairs"],
                            "qa_pairs": data["qa_pairs"]
                        }
