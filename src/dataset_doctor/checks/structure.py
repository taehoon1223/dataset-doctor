"""① File-level structure: missing/broken files, duplicates, split leaks."""
from __future__ import annotations

from collections import defaultdict

from dataset_doctor.core.models import EMPTY_FILE, Dataset, Issue, Sample, Severity
from dataset_doctor.core.registry import register_check


@register_check(stage="structure")
def image_file(sample: Sample):
    if sample.image_error == EMPTY_FILE:
        yield Issue("empty_file", Severity.ERROR, sample.image, sample.label_file, None, {"file": "image"})
    elif sample.image_error:
        yield Issue("corrupt_image", Severity.ERROR, sample.image, sample.label_file, None, {"error": sample.image_error})


@register_check(stage="structure")
def label_presence(sample: Sample):
    if sample.label_file is None:
        yield Issue("image_without_label", Severity.INFO, sample.image, None, None)
    elif not sample.annotations and not sample.syntax_errors:
        # An empty label file is how Ultralytics marks a background image, so this is informational.
        yield Issue("empty_file", Severity.INFO, sample.image, sample.label_file, None, {"file": "label"})


@register_check(stage="structure", scope="dataset")
def orphan_labels(dataset: Dataset):
    for label in dataset.orphan_labels:
        yield Issue("label_without_image", Severity.ERROR, None, label, None)


@register_check(stage="structure", scope="dataset")
def duplicate_images(dataset: Dataset):
    """Same file hash twice in one split -> duplicate_image; across splits -> split_leak."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in dataset.samples:
        if sample.file_hash:
            groups[sample.file_hash].append(sample)

    order = {split: i for i, split in enumerate(dataset.splits)}
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda s: (order[s.split], s.image))
        first_in_split: dict[str, Sample] = {}
        for sample in group:
            first = first_in_split.setdefault(sample.split, sample)
            if first is not sample:
                yield Issue("duplicate_image", Severity.WARNING, sample.image, sample.label_file, None,
                            {"duplicate_of": first.image})
        splits = list(first_in_split)
        for split in splits[1:]:
            sample = first_in_split[split]
            yield Issue("split_leak", Severity.WARNING, sample.image, sample.label_file, None,
                        {"splits": [splits[0], split], "same_as": first_in_split[splits[0]].image})
