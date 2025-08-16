"""Intelligent Caching System with ML-based Cache Eviction and Prefetching."""

import logging
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .monitoring_integration import get_monitoring_integration

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    computation_time: float = 0.0
    priority_score: float = 0.0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0


class PredictiveCache:
    """Intelligent cache with ML-based eviction and prefetching."""

    def __init__(self,
                 max_size_mb: int = 100,
                 max_entries: int = 1000,
                 ttl_seconds: int = 3600,
                 enable_ml_predictions: bool = True):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enable_ml_predictions = enable_ml_predictions

        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.stats = CacheStats()

        # ML components
        self.access_predictor: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.feature_history: List[List[float]] = []
        self.access_history: List[float] = []

        # Access pattern tracking
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prefetch_candidates: Dict[str, float] = {}

        # Monitoring integration
        self.monitoring = get_monitoring_integration()

        # Background cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

        if self.enable_ml_predictions:
            self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background tasks for ML training and cleanup."""
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._background_tasks, daemon=True)
        self._cleanup_thread.start()

    def _background_tasks(self) -> None:
        """Background task for cleanup and ML training."""
        while self._running:
            try:
                # Periodic cleanup
                self._cleanup_expired()

                # Train ML model if we have enough data
                if len(self.feature_history) >= 50:
                    self._train_access_predictor()

                # Update prefetch candidates
                self._update_prefetch_candidates()

                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Background task error: {e}")
                time.sleep(60)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(key)
                    self.stats.misses += 1
                    self.monitoring.record_cache_miss()
                    return default

                # Update access statistics
                entry.last_accessed = datetime.now()
                entry.access_count += 1

                # Move to end (LRU)
                self._cache.move_to_end(key)

                # Record access pattern
                self.access_patterns[key].append(datetime.now())

                self.stats.hits += 1
                self.monitoring.record_cache_hit()

                return entry.value
            else:
                self.stats.misses += 1
                self.monitoring.record_cache_miss()
                return default

    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Store value in cache."""
        with self._lock:
            # Calculate value size
            size_bytes = self._calculate_size(value)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                computation_time=computation_time
            )

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Ensure we have space
            self._ensure_capacity(size_bytes)

            # Add to cache
            self._cache[key] = entry
            self.stats.size_bytes += size_bytes
            self.stats.entry_count += 1

            # Calculate priority score
            entry.priority_score = self._calculate_priority_score(entry)

            # Record access pattern
            self.access_patterns[key].append(datetime.now())

            # Update ML training data
            if self.enable_ml_predictions:
                self._record_access_features(entry)

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            else:
                return 1024  # Default estimate

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - entry.created_at).total_seconds() > self.ttl_seconds

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            del self._cache[key]

    def _ensure_capacity(self, additional_size: int) -> None:
        """Ensure cache has capacity for additional data."""
        # Check size limit
        while (self.stats.size_bytes + additional_size > self.max_size_bytes and
               self._cache):
            self._evict_entry()

        # Check entry count limit
        while len(self._cache) >= self.max_entries and self._cache:
            self._evict_entry()

    def _evict_entry(self) -> None:
        """Evict an entry using intelligent strategy."""
        if not self._cache:
            return

        if self.enable_ml_predictions and self.access_predictor is not None:
            # ML-based eviction
            victim_key = self._predict_eviction_candidate()
        else:
            # Fallback to LRU with priority scoring
            victim_key = self._select_lru_victim()

        if victim_key:
            self._remove_entry(victim_key)
            self.stats.evictions += 1
            logger.debug(f"Evicted cache entry: {victim_key}")

    def _predict_eviction_candidate(self) -> Optional[str]:
        """Use ML to predict best eviction candidate."""
        try:
            candidates = []
            features = []

            for key, entry in self._cache.items():
                feature_vector = self._extract_features(entry)
                predicted_next_access = self.access_predictor.predict([feature_vector])[0]

                candidates.append((key, predicted_next_access))
                features.append(feature_vector)

            # Sort by predicted access time (descending = least likely to be accessed soon)
            candidates.sort(key=lambda x: x[1], reverse=True)

            return candidates[0][0] if candidates else None

        except Exception as e:
            logger.error(f"ML eviction prediction failed: {e}")
            return self._select_lru_victim()

    def _select_lru_victim(self) -> Optional[str]:
        """Select victim using LRU with priority scoring."""
        if not self._cache:
            return None

        # Calculate scores for all entries
        candidates = []
        for key, entry in self._cache.items():
            score = self._calculate_eviction_score(entry)
            candidates.append((key, score))

        # Sort by score (ascending = best candidates for eviction)
        candidates.sort(key=lambda x: x[1])

        return candidates[0][0] if candidates else next(iter(self._cache))

    def _calculate_priority_score(self, entry: CacheEntry) -> float:
        """Calculate priority score for cache entry."""
        age_hours = (datetime.now() - entry.created_at).total_seconds() / 3600
        access_frequency = entry.access_count / max(age_hours, 0.1)
        computation_value = min(entry.computation_time, 10.0) / 10.0  # Normalize to 0-1

        # Higher score = higher priority
        score = (access_frequency * 0.4 +
                computation_value * 0.3 +
                (1.0 / max(age_hours, 0.1)) * 0.3)

        return score

    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score (lower = better candidate for eviction)."""
        return -self._calculate_priority_score(entry)  # Invert priority

    def _extract_features(self, entry: CacheEntry) -> List[float]:
        """Extract features for ML prediction."""
        now = datetime.now()
        age_hours = (now - entry.created_at).total_seconds() / 3600
        time_since_access = (now - entry.last_accessed).total_seconds() / 3600

        # Access pattern features
        access_times = self.access_patterns.get(entry.key, [])
        recent_accesses = len([t for t in access_times if (now - t).total_seconds() < 3600])
        access_frequency = len(access_times) / max(age_hours, 0.1)

        # Time-based features
        hour_of_day = now.hour
        day_of_week = now.weekday()

        features = [
            age_hours,
            time_since_access,
            entry.access_count,
            recent_accesses,
            access_frequency,
            entry.size_bytes / 1024,  # Size in KB
            entry.computation_time,
            hour_of_day,
            day_of_week,
            entry.priority_score
        ]

        return features

    def _record_access_features(self, entry: CacheEntry) -> None:
        """Record features for ML training."""
        if not self.enable_ml_predictions:
            return

        features = self._extract_features(entry)

        # Simulate "time to next access" as target variable
        # In a real scenario, this would be collected over time
        target = np.random.exponential(scale=2.0)  # Placeholder

        self.feature_history.append(features)
        self.access_history.append(target)

        # Keep only recent history
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-1000:]
            self.access_history = self.access_history[-1000:]

    def _train_access_predictor(self) -> None:
        """Train ML model to predict access patterns."""
        try:
            if len(self.feature_history) < 50:
                return

            X = np.array(self.feature_history)
            y = np.array(self.access_history)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.access_predictor = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.access_predictor.fit(X_scaled, y)

            logger.info("Cache access predictor trained successfully")

        except Exception as e:
            logger.error(f"Failed to train access predictor: {e}")

    def _update_prefetch_candidates(self) -> None:
        """Update prefetch candidates based on access patterns."""
        now = datetime.now()

        for key, access_times in self.access_patterns.items():
            if len(access_times) < 2:
                continue

            # Calculate access intervals
            intervals = []
            for i in range(1, len(access_times)):
                interval = (access_times[i] - access_times[i-1]).total_seconds()
                intervals.append(interval)

            if intervals:
                avg_interval = np.mean(intervals)
                last_access = access_times[-1]
                time_since_last = (now - last_access).total_seconds()

                # Predict next access time
                if time_since_last >= avg_interval * 0.8:
                    prediction_confidence = min(1.0, time_since_last / avg_interval)
                    self.prefetch_candidates[key] = prediction_confidence

    def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

    def get_prefetch_candidates(self) -> List[Tuple[str, float]]:
        """Get candidates for prefetching sorted by confidence."""
        candidates = list(self.prefetch_candidates.items())
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:10]  # Top 10 candidates

    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate_percent": hit_rate,
                "evictions": self.stats.evictions,
                "size_mb": self.stats.size_bytes / (1024 * 1024),
                "entry_count": self.stats.entry_count,
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "max_entries": self.max_entries,
                "ml_enabled": self.enable_ml_predictions,
                "prefetch_candidates": len(self.prefetch_candidates)
            }

    def export_metrics(self) -> str:
        """Export cache metrics in Prometheus format."""
        stats = self.get_stats()

        metrics = [
            "# HELP docker_optimizer_cache_hits_total Total cache hits",
            "# TYPE docker_optimizer_cache_hits_total counter",
            f"docker_optimizer_cache_hits_total {stats['hits']}",
            "",
            "# HELP docker_optimizer_cache_misses_total Total cache misses",
            "# TYPE docker_optimizer_cache_misses_total counter",
            f"docker_optimizer_cache_misses_total {stats['misses']}",
            "",
            "# HELP docker_optimizer_cache_hit_rate_percent Cache hit rate percentage",
            "# TYPE docker_optimizer_cache_hit_rate_percent gauge",
            f"docker_optimizer_cache_hit_rate_percent {stats['hit_rate_percent']}",
            "",
            "# HELP docker_optimizer_cache_size_bytes Current cache size in bytes",
            "# TYPE docker_optimizer_cache_size_bytes gauge",
            f"docker_optimizer_cache_size_bytes {self.stats.size_bytes}",
            "",
            "# HELP docker_optimizer_cache_entries Current number of cache entries",
            "# TYPE docker_optimizer_cache_entries gauge",
            f"docker_optimizer_cache_entries {stats['entry_count']}"
        ]

        return "\n".join(metrics)

    def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)


class CacheManager:
    """Manages multiple cache instances with different strategies."""

    def __init__(self):
        # Different caches for different use cases
        self.caches = {
            "optimization_results": PredictiveCache(max_size_mb=50, ttl_seconds=1800),
            "security_scans": PredictiveCache(max_size_mb=30, ttl_seconds=3600),
            "image_analysis": PredictiveCache(max_size_mb=20, ttl_seconds=7200),
            "registry_data": PredictiveCache(max_size_mb=10, ttl_seconds=1800)
        }

    def get_cache(self, cache_name: str) -> Optional[PredictiveCache]:
        """Get specific cache instance."""
        return self.caches.get(cache_name)

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        overall_stats = {
            "total_hits": 0,
            "total_misses": 0,
            "total_size_mb": 0.0,
            "total_entries": 0,
            "cache_details": {}
        }

        for name, cache in self.caches.items():
            stats = cache.get_stats()
            overall_stats["total_hits"] += stats["hits"]
            overall_stats["total_misses"] += stats["misses"]
            overall_stats["total_size_mb"] += stats["size_mb"]
            overall_stats["total_entries"] += stats["entry_count"]
            overall_stats["cache_details"][name] = stats

        total_requests = overall_stats["total_hits"] + overall_stats["total_misses"]
        overall_stats["overall_hit_rate"] = (
            overall_stats["total_hits"] / total_requests * 100
            if total_requests > 0 else 0.0
        )

        return overall_stats

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()

    def stop_all(self) -> None:
        """Stop all cache background tasks."""
        for cache in self.caches.values():
            cache.stop()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_cache(cache_name: str) -> Optional[PredictiveCache]:
    """Get specific cache instance."""
    manager = get_cache_manager()
    return manager.get_cache(cache_name)


def cache_optimization_result(dockerfile_hash: str, result: Any, computation_time: float = 0.0) -> None:
    """Cache optimization result."""
    cache = get_cache("optimization_results")
    if cache:
        cache.put(dockerfile_hash, result, computation_time)


def get_cached_optimization(dockerfile_hash: str) -> Any:
    """Get cached optimization result."""
    cache = get_cache("optimization_results")
    if cache:
        return cache.get(dockerfile_hash)
    return None


def clear_all_caches() -> None:
    """Clear all caches."""
    manager = get_cache_manager()
    manager.clear_all()


def get_cache_stats() -> Dict[str, Any]:
    """Get overall cache statistics."""
    manager = get_cache_manager()
    return manager.get_overall_stats()
