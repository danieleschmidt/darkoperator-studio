"""Intelligent caching system for models and computation results."""

import torch
import pickle
import hashlib
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from collections import OrderedDict
import logging
import json


logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove oldest item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


class ModelCache:
    """Intelligent model caching with memory management."""
    
    def __init__(self, cache_dir: str = "./cache/models", max_memory_gb: float = 4.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        
        # In-memory cache for frequently used models
        self.memory_cache = LRUCache(max_size=5)
        
        # Disk cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        self.lock = threading.RLock()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _compute_key(self, model_name: str, config: Dict[str, Any]) -> str:
        """Compute cache key for model configuration."""
        config_str = json.dumps(config, sort_keys=True)
        key_data = f"{model_name}:{config_str}".encode()
        return hashlib.sha256(key_data).hexdigest()[:16]
    
    def _estimate_model_size(self, model: torch.nn.Module) -> int:
        """Estimate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size
    
    def get(self, model_name: str, config: Dict[str, Any]) -> Optional[torch.nn.Module]:
        """Get cached model."""
        cache_key = self._compute_key(model_name, config)
        
        with self.lock:
            # Check memory cache first
            model = self.memory_cache.get(cache_key)
            if model is not None:
                logger.debug(f"Model cache hit (memory): {model_name}")
                return model
            
            # Check disk cache
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                try:
                    model = torch.load(cache_path, map_location='cpu')
                    
                    # Add to memory cache if it fits
                    model_size = self._estimate_model_size(model)
                    if model_size < self.max_memory_bytes // 5:  # Use at most 20% per model
                        self.memory_cache.put(cache_key, model)
                    
                    # Update access time
                    self.metadata[cache_key] = {
                        'model_name': model_name,
                        'config': config,
                        'last_access': time.time(),
                        'access_count': self.metadata.get(cache_key, {}).get('access_count', 0) + 1,
                        'size_bytes': model_size
                    }
                    self._save_metadata()
                    
                    logger.debug(f"Model cache hit (disk): {model_name}")
                    return model
                    
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}")
                    cache_path.unlink(missing_ok=True)
            
            logger.debug(f"Model cache miss: {model_name}")
            return None
    
    def put(self, model_name: str, config: Dict[str, Any], model: torch.nn.Module) -> None:
        """Cache model."""
        cache_key = self._compute_key(model_name, config)
        model_size = self._estimate_model_size(model)
        
        with self.lock:
            # Always add to memory cache if it fits
            if model_size < self.max_memory_bytes // 5:
                self.memory_cache.put(cache_key, model)
            
            # Save to disk
            cache_path = self.cache_dir / f"{cache_key}.pt"
            try:
                torch.save(model, cache_path)
                
                # Update metadata
                self.metadata[cache_key] = {
                    'model_name': model_name,
                    'config': config,
                    'cached_time': time.time(),
                    'last_access': time.time(),
                    'access_count': 1,
                    'size_bytes': model_size
                }
                self._save_metadata()
                
                logger.info(f"Cached model: {model_name} ({model_size / (1024**2):.1f} MB)")
                
                # Clean up old cache entries if needed
                self._cleanup_cache()
                
            except Exception as e:
                logger.error(f"Failed to cache model {model_name}: {e}")
    
    def _cleanup_cache(self):
        """Clean up old cache entries to stay within memory limits."""
        total_size = sum(meta.get('size_bytes', 0) for meta in self.metadata.values())
        
        if total_size > self.max_memory_bytes:
            # Sort by last access time (oldest first)
            items = [(key, meta) for key, meta in self.metadata.items()]
            items.sort(key=lambda x: x[1].get('last_access', 0))
            
            # Remove oldest entries until under limit
            for cache_key, meta in items:
                if total_size <= self.max_memory_bytes * 0.8:  # Leave 20% buffer
                    break
                
                cache_path = self.cache_dir / f"{cache_key}.pt"
                cache_path.unlink(missing_ok=True)
                
                size_freed = meta.get('size_bytes', 0)
                total_size -= size_freed
                
                del self.metadata[cache_key]
                logger.info(f"Removed cached model: {meta.get('model_name', cache_key)} ({size_freed / (1024**2):.1f} MB freed)")
            
            self._save_metadata()
    
    def clear(self):
        """Clear all cached models."""
        with self.lock:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pt"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
            logger.info("Model cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(meta.get('size_bytes', 0) for meta in self.metadata.values())
        
        return {
            'memory_cache': self.memory_cache.stats(),
            'disk_cache_entries': len(self.metadata),
            'total_size_mb': total_size / (1024**2),
            'max_size_mb': self.max_memory_bytes / (1024**2),
            'usage_percent': (total_size / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0
        }


class ResultCache:
    """Cache for expensive computation results."""
    
    def __init__(self, cache_dir: str = "./cache/results", ttl_hours: float = 24.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        
        # In-memory cache for recent results
        self.memory_cache = LRUCache(max_size=100)
        
        self.lock = threading.RLock()
    
    def _compute_key(self, inputs: Any) -> str:
        """Compute cache key for inputs."""
        # Serialize inputs to create hash
        if torch.is_tensor(inputs):
            # For tensors, use shape and a sample of values
            if inputs.numel() > 1000:
                # For large tensors, sample values to avoid expensive hashing
                sample_indices = torch.linspace(0, inputs.numel()-1, 100, dtype=torch.long)
                sample_values = inputs.flatten()[sample_indices]
                key_data = f"tensor_{inputs.shape}_{sample_values.sum().item()}"
            else:
                key_data = f"tensor_{inputs.shape}_{inputs.sum().item()}"
        else:
            # For other types, use string representation
            key_data = str(inputs)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, inputs: Any) -> Optional[Any]:
        """Get cached result."""
        cache_key = self._compute_key(inputs)
        
        with self.lock:
            # Check memory cache first
            result = self.memory_cache.get(cache_key)
            if result is not None:
                return result
            
            # Check disk cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    # Check if cache entry is still valid
                    cache_time = cache_path.stat().st_mtime
                    if time.time() - cache_time < self.ttl_seconds:
                        with open(cache_path, 'rb') as f:
                            result = pickle.load(f)
                        
                        # Add to memory cache
                        self.memory_cache.put(cache_key, result)
                        return result
                    else:
                        # Cache expired, remove file
                        cache_path.unlink()
                
                except Exception as e:
                    logger.warning(f"Failed to load cached result: {e}")
                    cache_path.unlink(missing_ok=True)
        
        return None
    
    def put(self, inputs: Any, result: Any) -> None:
        """Cache computation result."""
        cache_key = self._compute_key(inputs)
        
        with self.lock:
            # Add to memory cache
            self.memory_cache.put(cache_key, result)
            
            # Save to disk
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                
                logger.debug(f"Cached result: {cache_key}")
                
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                if current_time - cache_file.stat().st_mtime > self.ttl_seconds:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to check cache file {cache_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")
    
    def clear(self):
        """Clear all cached results."""
        with self.lock:
            self.memory_cache.clear()
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            logger.info("Result cache cleared")