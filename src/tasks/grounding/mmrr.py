import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig
from rhoknp import Document, Sentence

from tasks.util import FileBasedResourceManagerMixin
from utils.annotation import ImageTextAnnotation, SentenceAnnotation, UtteranceAnnotation
from utils.mmrr import MMRefRelPrediction
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo


class MultiModalReferenceRelationGrounding(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document_path: Annotated[Path, luigi.Parameter()] = luigi.PathParameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(parents=True, exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self):
        if (gpu_id := self.acquire_resource()) is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            prediction = run_mmrr(
                self.cfg,
                dataset_dir=self.dataset_dir,
                document_path=self.document_path,
                env=env,
            )
            with self.output().open(mode="w") as f:
                f.write(prediction.to_json(ensure_ascii=False, indent=2))
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise e
        finally:
            self.release_resource(gpu_id)


def split_utterances_to_sentences(
    info: DatasetInfo, annotation: ImageTextAnnotation, document: Document
) -> list[SentenceAnnotation]:
    # collect sids corresponding to utterances
    sid_mapper = [utt.sids for utt in info.utterances]
    assert len(sid_mapper) == len(annotation.utterances)

    # split utterances field
    vis_sentences = []  # [{"sid": xx, "phrases": xx}, ...]
    for idx, utterance in enumerate(annotation.utterances):
        assert isinstance(utterance, UtteranceAnnotation)
        sids = sid_mapper[idx]
        vis_sentences.extend([SentenceAnnotation(text="", phrases=utterance.phrases, sid=sid) for sid in sids])
    assert len(vis_sentences) == len(document.sentences)

    # format visual phrases by base phrases
    for sentence in document.sentences:
        s_idx = int(sentence.sid.split("-")[-1])  # '<did>-1-<idx>' -> idx
        vis_sentence = vis_sentences[s_idx]
        vis_sentence.text = sentence.text
        if len(sentence.base_phrases) != len(vis_sentence.phrases):
            # update visual phrase annotation
            doc_p = [b.text for b in sentence.base_phrases]
            vis_p = [u.text for u in vis_sentence.phrases]
            start_idx = vis_p.index(doc_p[0])
            end_idx = start_idx + len(doc_p)
            vis_sentence.phrases = vis_sentence.phrases[start_idx:end_idx]
    return vis_sentences


def run_mmrr(cfg: DictConfig, dataset_dir: Path, document_path: Path, env: dict[str, str]) -> PhraseGroundingPrediction:
    dataset_info = DatasetInfo.from_json(dataset_dir.joinpath("info.json").read_text())
    document = Document.from_knp(document_path.read_text())
    gold_annotation = ImageTextAnnotation.from_json(
        Path(cfg.gold_annotation_dir).joinpath(f"{dataset_info.scenario_id}.json").read_text()
    )
    vis_sentences = split_utterances_to_sentences(
        info=dataset_info,
        annotation=gold_annotation,
        document=document,
    )
    sid2vis_sentence: dict[str, SentenceAnnotation] = {}
    for sentence in vis_sentences:
        assert isinstance(sentence, SentenceAnnotation)
        sid2vis_sentence.update({sentence.sid: sentence})
    sid2doc_sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}

    utterance_predictions: list[UtterancePrediction] = []
    for idx, utterance in enumerate(dataset_info.utterances):
        if idx >= 1:
            prev_utterance = dataset_info.utterances[idx - 1]
            start_index = math.ceil(prev_utterance.end / 1000)
        else:
            start_index = 0
        if idx + 1 < len(dataset_info.utterances):
            next_utterance = dataset_info.utterances[idx + 1]
            end_index = math.ceil(next_utterance.start / 1000)
        else:
            end_index = len(dataset_info.images)
        corresponding_images = dataset_info.images[start_index:end_index]
        utterances_in_window = dataset_info.utterances[max(0, idx + 1 - cfg.num_utterances_in_window) : idx + 1]
        sentence_ids = [sid for utterance in utterances_in_window for sid in utterance.sids]
        doc_utterance = Document.from_sentences([sid2doc_sentence[sid] for sid in utterance.sids])
        phrases: list[PhrasePrediction] = [
            PhrasePrediction(
                sid=base_phrase.sentence.sid,
                index=base_phrase.global_index,
                text=base_phrase.text,
                relations=[],
            )
            for base_phrase in doc_utterance.base_phrases
        ]

        # Skip empty utterance
        if len(utterance.sids) == 0:
            utterance_predictions.append(
                UtterancePrediction(text=doc_utterance.text, sids=utterance.sids, phrases=phrases)
            )
            continue

        with tempfile.TemporaryDirectory() as root_dir:
            for image_idx in range(start_index, end_index):
                utterances = [
                    UtteranceAnnotation(text=sid2vis_sentence[sid].text, phrases=sid2vis_sentence[sid].phrases)
                    for sid in sentence_ids
                ]
                frame_annotation = ImageTextAnnotation(
                    scenario_id=dataset_info.scenario_id,
                    images=[gold_annotation.images[image_idx]],
                    utterances=utterances,
                )
                target = Path(root_dir) / f"{dataset_info.scenario_id}-{image_idx}.json"
                target.write_text(frame_annotation.to_json(ensure_ascii=False, indent=2))
            shutil.copy(document_path, root_dir)

            out_dir = Path(root_dir) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/scripts/run_mmref.py",
                    f"checkpoint={cfg.checkpoint}",
                    f"input_dir={root_dir}",
                    f"export_dir={out_dir}",
                    f"object_file_root={cfg.object_file_root}",
                    f"object_file_name={cfg.object_file_name}",
                    "num_workers=0",
                    "devices=1",
                ],
                check=True,
                env=env,
            )
            predictions = [
                MMRefRelPrediction.from_json(file.read_text()) for file in sorted(Path(out_dir).glob("*.json"))
            ]

        assert len(corresponding_images) == len(predictions), f"{len(corresponding_images)} != {len(predictions)}"
        for image, prediction in zip(corresponding_images, predictions):
            for phrase in phrases:
                phrase_prediction = prediction.phrases[phrase.index + len(prediction.phrases) - len(phrases)]
                assert phrase_prediction.text == phrase.text
                for relation in phrase_prediction.relations:
                    for bounding_box in relation.bounding_boxes:
                        assert bounding_box.image_id == image.id
                        phrase.relations.append(
                            RelationPrediction(
                                type=relation.type,
                                image_id=image.id,
                                bounding_box=BoundingBoxPrediction(
                                    image_id=image.id,
                                    rect=bounding_box.rect,
                                    confidence=bounding_box.confidence,
                                ),
                            )
                        )
        utterance_predictions.append(UtterancePrediction(text=doc_utterance.text, sids=utterance.sids, phrases=phrases))

    return PhraseGroundingPrediction(
        scenario_id=dataset_info.scenario_id, images=dataset_info.images, utterances=utterance_predictions
    )
