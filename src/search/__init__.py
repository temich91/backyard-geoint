"""Поиск мест по готовым векторам, независимо от модели извлечения признаков."""

from .index import PlaceIndex, SearchResult, load_embeddings, read_metadata

__all__ = ["PlaceIndex", "SearchResult", "load_embeddings", "read_metadata"]
