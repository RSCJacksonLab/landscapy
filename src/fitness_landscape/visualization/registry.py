from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

from ..core.annotation import AnnotationLayer


@dataclass(slots=True)
class AnnotationDescriptor:
    """
    Metadata describing an annotation scheme available to the visualization layer.
    """

    name: str
    layer: AnnotationLayer
    source: str = "landscape"
    palette_key: Optional[str] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


class AnnotationRegistry:
    """
    Lightweight registry tracking annotation schemes and their descriptors.
    """

    def __init__(self) -> None:
        self._items: Dict[str, AnnotationDescriptor] = {}

    def register(
        self,
        name: str,
        layer: AnnotationLayer,
        *,
        source: str = "landscape",
        palette_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AnnotationDescriptor:
        if name in self._items:
            raise ValueError(f"Annotation '{name}' is already registered.")
        descriptor = AnnotationDescriptor(
            name=name,
            layer=layer,
            source=source,
            palette_key=palette_key,
            metadata=dict(metadata) if metadata else {},
        )
        self._items[name] = descriptor
        return descriptor

    def update_palette(self, name: str, palette_key: str) -> None:
        if name not in self._items:
            raise KeyError(f"Annotation '{name}' is not registered.")
        self._items[name].palette_key = palette_key

    def get(self, name: str) -> AnnotationDescriptor:
        if name not in self._items:
            raise KeyError(f"Annotation '{name}' is not registered.")
        return self._items[name]

    def discard(self, name: str) -> None:
        self._items.pop(name, None)

    def items(self):
        return self._items.items()

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


class PaletteStore:
    """
    Simple palette cache keyed by annotation or scheme identifiers.
    """

    def __init__(self) -> None:
        self._palettes: Dict[str, Any] = {}

    def register_palette(self, key: str, palette: Any) -> None:
        self._palettes[key] = palette

    def get_palette(self, key: str | None) -> Any | None:
        if key is None:
            return None
        return self._palettes.get(key)

    def has_palette(self, key: str) -> bool:
        return key in self._palettes

    def clear(self) -> None:
        self._palettes.clear()
