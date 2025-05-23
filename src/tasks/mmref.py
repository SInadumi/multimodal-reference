import copy
import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Annotated, Literal, Optional, Union

import luigi
from omegaconf import DictConfig
from rhoknp import BasePhrase, Document, Sentence
from rhoknp.cohesion import EndophoraArgument, EntityManager, RelMode, RelTagList

from tasks import (
    CohesionAnalysis,
    DeticPhraseGrounding,
    GLIPPhraseGrounding,
    MDETRPhraseGrounding,
    MultiModalReferenceRelationGrounding,
    MultipleObjectTracking,
    SoMPhraseGrounding,
)
from utils.annotation import BoundingBox as BoundingBoxAnnotation
from utils.annotation import ImageAnnotation, ImageTextAnnotation
from utils.mot import DetectionLabels
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import PhraseGroundingPrediction, RelationPrediction
from utils.util import DatasetInfo, box_iou

PHRASE_GROUNDING_MODEL_MAP: dict[str, type[luigi.Task]] = {
    "glip": GLIPPhraseGrounding,
    "mdetr": MDETRPhraseGrounding,
    "detic": DeticPhraseGrounding,
    "mmrr": MultiModalReferenceRelationGrounding,
    "som": SoMPhraseGrounding,
}


class MultimodalReference(luigi.Task):
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_dir = Path(self.cfg.dataset_dir) / self.scenario_id
        self.prediction_dir = Path(self.cfg.prediction_dir)
        self.gold_document_path = Path(self.cfg.gold_knp_dir) / f"{self.scenario_id}.knp"

    def requires(self) -> dict[str, luigi.Task]:
        tasks: dict[str, luigi.Task] = {
            "cohesion": CohesionAnalysis(
                cfg=self.cfg.cohesion, scenario_id=self.scenario_id, dataset_dir=self.dataset_dir
            ),
            "grounding": PHRASE_GROUNDING_MODEL_MAP[self.cfg.phrase_grounding_model](
                cfg=getattr(self.cfg, self.cfg.phrase_grounding_model),
                scenario_id=self.scenario_id,
                document_path=self.gold_document_path,
                dataset_dir=self.dataset_dir,
            ),
        }
        if self.cfg.mot_relax_mode == "pred":
            tasks["mot"] = MultipleObjectTracking(cfg=self.cfg.mot, scenario_id=self.scenario_id)
        return tasks

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.prediction_dir.joinpath(f"{self.scenario_id}.json"))

    def complete(self) -> bool:
        if not Path(self.output().path).exists():
            return False

        self_mtime = Path(self.output().path).stat().st_mtime
        for task in self.requires().values():
            if not task.complete():
                return False
            output = task.output()
            assert isinstance(output, luigi.LocalTarget), f"output is not LocalTarget: {output}"
            if Path(output.path).stat().st_mtime > self_mtime:
                return False

        return True

    def run(self):
        gold_document = Document.from_knp(self.gold_document_path.read_text())
        with self.input()["cohesion"].open(mode="r") as f:
            cohesion_prediction = Document.from_knp(f.read())
        with self.input()["grounding"].open(mode="r") as f:
            phrase_grounding_prediction = PhraseGroundingPrediction.from_json(f.read())
        mot_prediction: Optional[DetectionLabels] = None
        if self.cfg.mot_relax_mode == "pred":
            with self.input()["mot"].open(mode="r") as f:
                mot_prediction = DetectionLabels.from_json(f.read())
        gold_annotation = ImageTextAnnotation.from_json(
            Path(self.cfg.gold_annotation_dir).joinpath(f"{self.scenario_id}.json").read_text()
        )
        prediction = run_prediction(
            cohesion_prediction=cohesion_prediction,
            phrase_grounding_prediction=phrase_grounding_prediction,
            mot_prediction=mot_prediction,
            gold_document=gold_document,
            image_annotations=gold_annotation.images,
            coref_relax_mode=self.cfg.coref_relax_mode,
            mot_relax_mode=self.cfg.mot_relax_mode,
            rel_types=self.cfg.rel_types,
            confidence_modification_method=self.cfg.confidence_modification_method,
            dataset_info=DatasetInfo.from_json(self.dataset_dir.joinpath("info.json").read_text()),
        )
        with self.output().open(mode="w") as f:
            f.write(prediction.to_json(ensure_ascii=False, indent=2))


def run_prediction(
    cohesion_prediction: Document,
    phrase_grounding_prediction: PhraseGroundingPrediction,
    mot_prediction: Optional[DetectionLabels],
    gold_document: Document,
    image_annotations: list[ImageAnnotation],
    coref_relax_mode: Optional[str],
    mot_relax_mode: Optional[str],
    rel_types: list[str],
    confidence_modification_method: Literal["max", "min", "mean"],
    dataset_info: DatasetInfo,
) -> PhraseGroundingPrediction:
    parsed_document = preprocess_document(cohesion_prediction)

    if coref_relax_mode == "pred":
        relax_prediction_with_coreference(phrase_grounding_prediction, parsed_document)
    elif coref_relax_mode == "gold":
        relax_prediction_with_coreference(phrase_grounding_prediction, gold_document)

    if mot_relax_mode == "pred":
        assert mot_prediction is not None
        relax_prediction_with_mot(
            phrase_grounding_prediction,
            mot_prediction,
            rel_types=rel_types,
            confidence_modification_method=confidence_modification_method,
            dataset_info=dataset_info,
        )
    elif mot_relax_mode == "gold":
        relax_prediction_with_mot(
            phrase_grounding_prediction,
            image_annotations,
            rel_types=rel_types,
            confidence_modification_method=confidence_modification_method,
            dataset_info=dataset_info,
        )

    if coref_relax_mode is not None and mot_relax_mode is not None:
        # prev_phrase_grounding_prediction = None
        count = 0
        while count < 3:
            # prev_phrase_grounding_prediction = copy.deepcopy(phrase_grounding_prediction)
            if coref_relax_mode == "pred":
                relax_prediction_with_coreference(phrase_grounding_prediction, parsed_document)
            elif coref_relax_mode == "gold":
                relax_prediction_with_coreference(phrase_grounding_prediction, gold_document)
            if mot_relax_mode == "pred":
                assert mot_prediction is not None
                relax_prediction_with_mot(
                    phrase_grounding_prediction,
                    mot_prediction,
                    rel_types=rel_types,
                    confidence_modification_method=confidence_modification_method,
                    dataset_info=dataset_info,
                )
            elif mot_relax_mode == "gold":
                relax_prediction_with_mot(
                    phrase_grounding_prediction,
                    image_annotations,
                    rel_types=rel_types,
                    confidence_modification_method=confidence_modification_method,
                    dataset_info=dataset_info,
                )
            count += 1

    mm_reference_prediction = relax_prediction_with_pas_bridging(phrase_grounding_prediction, parsed_document)

    # sort relations by confidence
    for utterance in mm_reference_prediction.utterances:
        for phrase in utterance.phrases:
            phrase.relations.sort(key=lambda rel: rel.bounding_box.confidence, reverse=True)
    return mm_reference_prediction


def relax_prediction_with_mot(
    phrase_grounding_prediction: PhraseGroundingPrediction,
    image_annotations: Union[list[ImageAnnotation], DetectionLabels],
    rel_types: list[str],
    confidence_modification_method: Literal["max", "min", "mean"],
    dataset_info: DatasetInfo,
) -> None:
    # create a bounding box cluster according to instance_id
    gold_bb_cluster: dict[str, list[BoundingBoxAnnotation]] = defaultdict(list)
    if isinstance(image_annotations, DetectionLabels):
        for idx in range(math.ceil(len(image_annotations.frames) / 30)):
            frame = image_annotations.frames[idx * 30]
            for bb in frame.bounding_boxes:
                gold_bb_cluster[str(bb.instance_id)].append(
                    BoundingBoxAnnotation(
                        image_id=f"{idx:03d}",
                        rect=bb.rect,
                        class_name=bb.class_name,
                        instance_id=str(bb.instance_id),
                    )
                )
    else:
        for image_annotation in image_annotations:
            for gold_bb in image_annotation.bounding_boxes:
                gold_bb_cluster[gold_bb.instance_id].append(gold_bb)

    image_id_to_bbs: dict[str, list[BoundingBoxAnnotation]] = defaultdict(list)
    for gold_bbs in gold_bb_cluster.values():
        for gold_bb in gold_bbs:
            image_id_to_bbs[gold_bb.image_id].append(gold_bb)

    assert len(phrase_grounding_prediction.utterances) == len(dataset_info.utterances)
    for idx, utterance_prediction in enumerate(phrase_grounding_prediction.utterances):
        if idx + 1 < len(dataset_info.utterances):
            next_utterance = dataset_info.utterances[idx + 1]
            end_index = math.ceil(next_utterance.start / 1000)
        else:
            end_index = len(dataset_info.images)
        images_before_in_utterance = dataset_info.images[:end_index]

        # フレーズ・格は独立
        for phrase_prediction, rel_type in itertools.product(utterance_prediction.phrases, rel_types):
            instance_id_to_relations: dict[str, list[RelationPrediction]] = defaultdict(list)
            relation_to_matched_bb: dict[RelationPrediction, BoundingBoxAnnotation] = {}
            # まずそれぞれの relation について最も IoU が高い MOT 由来の BB を割り当てる
            for relation in [rel for rel in phrase_prediction.relations if rel.type == rel_type]:
                bb_ious_in_frame = [
                    (bb, box_iou(relation.bounding_box.rect, bb.rect)) for bb in image_id_to_bbs[relation.image_id]
                ]
                if not bb_ious_in_frame:
                    continue
                # 最も IoU が高い gold_bb を探す
                gold_bb, iou = max(bb_ious_in_frame, key=lambda x: x[1])
                if iou < 0.5:
                    continue
                instance_id_to_relations[gold_bb.instance_id].append(relation)
                relation_to_matched_bb[relation] = gold_bb

            relation_to_modified_confidence: dict[RelationPrediction, float] = {}
            for instance_id, relations in instance_id_to_relations.items():
                relations_in_cluster: list[RelationPrediction] = relations[:]
                gold_bbs = gold_bb_cluster[instance_id]
                matched_bbs = [relation_to_matched_bb[rel] for rel in relations]
                unmatched_bbs = [bb for bb in gold_bbs if bb not in matched_bbs]
                # 現在か過去のBBのみ新規追加
                for image in images_before_in_utterance:
                    relations_in_cluster += [
                        RelationPrediction(
                            type=rel_type,
                            image_id=image.id,
                            bounding_box=BoundingBoxPrediction(
                                image_id=image.id,
                                rect=bb.rect,
                                confidence=-1.0,
                            ),
                        )
                        for bb in unmatched_bbs
                        if bb.image_id == image.id
                    ]
                for relation in relations_in_cluster:
                    # 自らと先行するフレームの関係のみ考える
                    preceding_relations = [
                        relation,
                        *[rel for rel in relations_in_cluster if rel.image_idx < relation.image_idx],
                    ]
                    # MOT 由来の関係を除外して元々付与されていた関係のみ考える
                    preceding_original_relations = [
                        rel for rel in preceding_relations if rel.bounding_box.confidence >= 0
                    ]
                    if not preceding_original_relations:
                        continue
                    # 先行するフレームに1つでも関係があれば，MOT由来の新しい関係を追加
                    if relation.bounding_box.confidence == -1.0:
                        phrase_prediction.relations.append(relation)

                    confidences = [rel.bounding_box.confidence for rel in preceding_original_relations]
                    if confidence_modification_method == "max":
                        modified_confidence = max(confidences)
                    elif confidence_modification_method == "min":
                        modified_confidence = min(confidences)
                    elif confidence_modification_method == "mean":
                        modified_confidence = mean(confidences)
                    else:
                        raise NotImplementedError
                    print(
                        f"{phrase_grounding_prediction.scenario_id}: {relation.image_id}: {gold_bbs[0].class_name}: confidence: {relation.bounding_box.confidence:.6f} -> {modified_confidence:.6f}"
                    )
                    relation_to_modified_confidence[relation] = modified_confidence
            # confidence の修正が他の関係の confidence の修正に影響しないよう一度に修正する
            for relation, modified_confidence in relation_to_modified_confidence.items():
                assert modified_confidence >= 0
                relation.bounding_box.confidence = modified_confidence


def relax_prediction_with_coreference(
    phrase_grounding_prediction: PhraseGroundingPrediction, document: Document
) -> None:
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}

    phrase_id_to_relations: dict[int, set[RelationPrediction]] = defaultdict(set)
    # convert phrase grounding result to phrase_id_to_relations
    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        assert len(base_phrases) == len(utterance.phrases)
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            phrase_id_to_relations[base_phrase.global_index].update(phrase_prediction.relations)

    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            coreferents: list[BasePhrase] = base_phrase.get_coreferents(include_nonidentical=False, include_self=False)
            new_relations: set[RelationPrediction] = set()
            for coreferent in coreferents:
                if coreferent.global_index < base_phrase.global_index:
                    new_relations.update(phrase_id_to_relations[coreferent.global_index])
            phrase_prediction.relations = sorted(
                set(phrase_prediction.relations) | new_relations, key=lambda r: r.image_id
            )


def relax_prediction_with_pas_bridging(
    phrase_grounding_prediction: PhraseGroundingPrediction,
    parsed_document: Document,
) -> PhraseGroundingPrediction:
    # TODO: 先行する基本句の関係のみを考慮する
    phrase_id_to_relations: dict[int, set[RelationPrediction]] = defaultdict(set)

    # convert phrase grounding result to phrase_id_to_relations
    sid2sentence = {sentence.sid: sentence for sentence in parsed_document.sentences}
    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            phrase_id_to_relations[base_phrase.global_index].update(phrase_prediction.relations)

    # relax annotation until convergence
    phrase_id_to_relations_prev: dict[int, set[RelationPrediction]] = {}
    while phrase_id_to_relations != phrase_id_to_relations_prev:
        phrase_id_to_relations_prev = copy.deepcopy(phrase_id_to_relations)
        relax_annotation_with_pas_bridging(parsed_document, phrase_id_to_relations)

    # convert phrase_id_to_relations to phrase grounding result
    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            relations = set(phrase_prediction.relations)
            relations.update(phrase_id_to_relations[base_phrase.global_index])
            phrase_prediction.relations = list(relations)

    return phrase_grounding_prediction


def relax_annotation_with_pas_bridging(
    document: Document, phrase_id_to_relations: dict[int, set[RelationPrediction]]
) -> None:
    for base_phrase in document.base_phrases:
        current_relations: set[RelationPrediction] = set()  # 述語を中心とした関係の集合
        current_relations.update(phrase_id_to_relations[base_phrase.global_index])
        new_relations: set[RelationPrediction] = set([])
        for case, arguments in base_phrase.pas.get_all_arguments(relax=False).items():
            argument_global_indices: set[int] = set()
            for argument in arguments:
                if isinstance(argument, EndophoraArgument):
                    argument_global_indices.add(argument.base_phrase.global_index)
            new_relations.update(
                {
                    RelationPrediction(type=case, image_id=rel.image_id, bounding_box=rel.bounding_box)
                    for argument_global_index in argument_global_indices
                    for rel in phrase_id_to_relations[argument_global_index]
                    if rel.type == "="
                }
            )
            # 格が一致する述語を中心とした関係の集合
            # relation の対象は argument_entity_ids と一致
            case_relations: set[RelationPrediction] = {rel for rel in current_relations if rel.type == case}
            for argument_global_index in argument_global_indices:
                phrase_id_to_relations[argument_global_index].update(
                    {
                        RelationPrediction(type="=", image_id=rel.image_id, bounding_box=rel.bounding_box)
                        for rel in case_relations
                    }
                )
        phrase_id_to_relations[base_phrase.global_index].update(new_relations)


def preprocess_document(document: Document) -> Document:
    for base_phrase in document.base_phrases:
        filtered = RelTagList()
        for rel_tag in base_phrase.rel_tags:
            # exclude '?' rel tags for simplicity
            if rel_tag.mode is RelMode.AMBIGUOUS and rel_tag.target != "なし":
                continue
            # exclude coreference relations of 用言
            # e.g., ...を[運んで]。[それ]が終わったら...
            if rel_tag.type == "=" and rel_tag.sid is not None:
                if target_base_phrase := base_phrase._get_target_base_phrase(rel_tag):
                    if ("体言" in base_phrase.features and "体言" in target_base_phrase.features) is False:
                        continue
            filtered.append(rel_tag)
        base_phrase.rel_tags = filtered
    document = document.reparse()
    # ensure that each base phrase has at least one entity
    for base_phrase in document.base_phrases:
        if len(base_phrase.entities) == 0:
            EntityManager.get_or_create_entity().add_mention(base_phrase)
    return document
