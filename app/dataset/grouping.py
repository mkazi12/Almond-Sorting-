"""Group related frames so one recording never straddles two splits (SCRUM-50).

A *group* is the smallest set of accepted images that must stay together:

  * images that share a ``source_lot``,
  * images that share a ``source_session`` (capture session / camera run),
  * images that share a ``near_duplicate_group_id``.

Those links are transitive. If image A and B share a lot, and B and C share a
session, then A, B and C are one group. This is what makes the split
leak-proof: no lot, no session and no near-duplicate cluster can end up on
both sides of a train/valid/test boundary, because each whole cluster is
assigned to a single partition.

``group_id`` is derived from the keys that make up the group, not from the
order images are read in, so rebuilding it from the same records gives the
same id. Adding a new image to an existing lot keeps the id. Adding an image
that links two existing groups merges them into one, and the merged group gets
a new id; that is expected and is why splits are frozen only after grouping.

An accepted image that is missing required provenance (see ``SourcePolicy``)
is not grouped. It is returned in ``GroupingResult.ungrouped`` with the
missing fields, so it is held out of splitting rather than guessed at.

This module does not move files, assign partitions, or hash anything.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.dataset.records import (
    AcceptedImage,
    SourcePolicy,
    missing_provenance_fields,
)

GROUP_ID_PREFIX = "grp-"
_GROUP_ID_HEX_LENGTH = 12

# Namespaces keep a lot called "A" from colliding with a session called "A".
LOT = "lot"
SESSION = "session"
NEAR_DUPLICATE = "dup"


class GroupingError(ValueError):
    """The records cannot be grouped."""


@dataclass(frozen=True)
class ImageGroup:
    group_id: str
    keys: tuple[str, ...]  # sorted "namespace:value" keys that define the group
    image_ids: tuple[str, ...]  # sorted

    @property
    def size(self) -> int:
        return len(self.image_ids)


@dataclass(frozen=True)
class UngroupedImage:
    image_id: str
    missing: tuple[str, ...]


@dataclass(frozen=True)
class GroupingResult:
    groups: tuple[ImageGroup, ...]  # sorted by group_id
    ungrouped: tuple[UngroupedImage, ...]  # sorted by image_id

    def group_of(self) -> dict[str, str]:
        """Map image_id -> group_id for every grouped image."""
        return {
            image_id: group.group_id
            for group in self.groups
            for image_id in group.image_ids
        }


def _image_keys(image: AcceptedImage) -> list[str]:
    keys: list[str] = []
    if image.source_lot is not None:
        keys.append(f"{LOT}:{image.source_lot}")
    if image.source_session is not None:
        keys.append(f"{SESSION}:{image.source_session}")
    if image.near_duplicate_group_id is not None:
        keys.append(f"{NEAR_DUPLICATE}:{image.near_duplicate_group_id}")
    return keys


def make_group_id(keys: Iterable[str]) -> str:
    """Stable id for a set of group keys, independent of their order."""
    joined = "\n".join(sorted(set(keys)))
    if not joined:
        raise GroupingError("A group needs at least one lot, session or near-duplicate key.")
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return GROUP_ID_PREFIX + digest[:_GROUP_ID_HEX_LENGTH]


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Always keep the smaller key as the root so results are deterministic.
            keep, drop = sorted((root_a, root_b))
            self._parent[drop] = keep


def assign_groups(images: Iterable[AcceptedImage], policy: SourcePolicy) -> GroupingResult:
    """Group accepted images by shared lot, session or near-duplicate id.

    Images missing a field required by ``policy`` are returned as ungrouped.
    An image that is eligible under the policy but has no key at all (for
    example when neither lot nor session is required and both are blank) is
    also ungrouped, because there is nothing to tie it to.
    """
    images = list(images)
    seen: set[str] = set()
    for image in images:
        if image.image_id in seen:
            raise GroupingError(f"Repeated image_id: {image.image_id}")
        seen.add(image.image_id)

    sets = _DisjointSet()
    keys_by_image: dict[str, list[str]] = {}
    ungrouped: list[UngroupedImage] = []

    for image in images:
        missing = missing_provenance_fields(image, policy)
        keys = _image_keys(image)
        if missing:
            ungrouped.append(UngroupedImage(image.image_id, missing))
            continue
        if not keys:
            ungrouped.append(UngroupedImage(image.image_id, ("lot_or_session",)))
            continue
        for key in keys:
            sets.add(key)
        for other in keys[1:]:
            sets.union(keys[0], other)
        keys_by_image[image.image_id] = keys

    members: dict[str, list[str]] = defaultdict(list)
    component_keys: dict[str, set[str]] = defaultdict(set)
    for image_id, keys in keys_by_image.items():
        root = sets.find(keys[0])
        members[root].append(image_id)
        component_keys[root].update(keys)

    groups = [
        ImageGroup(
            group_id=make_group_id(component_keys[root]),
            keys=tuple(sorted(component_keys[root])),
            image_ids=tuple(sorted(image_ids)),
        )
        for root, image_ids in members.items()
    ]
    groups.sort(key=lambda group: group.group_id)
    ungrouped.sort(key=lambda item: item.image_id)
    return GroupingResult(groups=tuple(groups), ungrouped=tuple(ungrouped))