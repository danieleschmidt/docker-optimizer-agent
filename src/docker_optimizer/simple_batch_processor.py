"""Simple batch processing for Docker Optimizer Agent."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .optimizer import DockerfileOptimizer
from .simple_metrics import SimpleMetricsCollector
from .models import OptimizationResult

logger = logging.getLogger(__name__)


class SimpleBatchProcessor:
    """Simple batch processing for multiple Dockerfiles."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.optimizer = DockerfileOptimizer()
        self.metrics = SimpleMetricsCollector()
    
    def process_dockerfiles(self, dockerfile_paths: List[str]) -> List[Tuple[str, OptimizationResult]]:
        """Process multiple Dockerfiles synchronously."""
        results = []
        start_time = time.time()
        
        logger.info(f"Starting batch processing of {len(dockerfile_paths)} Dockerfiles")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self._process_single_dockerfile, path): path
                for path in dockerfile_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                dockerfile_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append((dockerfile_path, result))
                    self.metrics.increment('dockerfiles_processed')
                    logger.info(f"Successfully processed: {dockerfile_path}")
                except Exception as e:
                    logger.error(f"Failed to process {dockerfile_path}: {e}")
                    self.metrics.increment('processing_errors')
        
        total_time = time.time() - start_time
        self.metrics.record_timing('batch_processing', total_time)
        self.metrics.set_gauge('last_batch_size', len(dockerfile_paths))
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        return results
    
    async def process_dockerfiles_async(self, dockerfile_paths: List[str]) -> List[Tuple[str, OptimizationResult]]:
        """Process multiple Dockerfiles asynchronously."""
        semaphore = asyncio.Semaphore(self.max_workers)
        start_time = time.time()
        
        async def process_with_semaphore(path):
            async with semaphore:
                return await self._process_single_dockerfile_async(path)
        
        logger.info(f"Starting async batch processing of {len(dockerfile_paths)} Dockerfiles")
        
        # Process all Dockerfiles concurrently
        tasks = [process_with_semaphore(path) for path in dockerfile_paths]
        results = []
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append((dockerfile_paths[i], result))
                self.metrics.increment('dockerfiles_processed')
            except Exception as e:
                logger.error(f"Failed to process {dockerfile_paths[i]}: {e}")
                self.metrics.increment('processing_errors')
        
        total_time = time.time() - start_time
        self.metrics.record_timing('async_batch_processing', total_time)
        
        logger.info(f"Async batch processing completed in {total_time:.2f}s")
        return results
    
    def _process_single_dockerfile(self, dockerfile_path: str) -> OptimizationResult:
        """Process a single Dockerfile."""
        start_time = time.time()
        
        try:
            path = Path(dockerfile_path)
            if not path.exists():
                raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
            
            content = path.read_text(encoding='utf-8')
            result = self.optimizer.optimize_dockerfile(content)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.record_timing('single_dockerfile_processing', processing_time)
            self.metrics.increment('security_fixes_applied', len(result.security_fixes))
            self.metrics.increment('layer_optimizations', len(result.layer_optimizations))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {dockerfile_path}: {e}")
            raise
    
    async def _process_single_dockerfile_async(self, dockerfile_path: str) -> OptimizationResult:
        """Process a single Dockerfile asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_single_dockerfile, dockerfile_path)
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'dockerfiles_processed': self.metrics.get_counter('dockerfiles_processed'),
            'processing_errors': self.metrics.get_counter('processing_errors'),
            'security_fixes_applied': self.metrics.get_counter('security_fixes_applied'),
            'layer_optimizations': self.metrics.get_counter('layer_optimizations'),
            'last_batch_size': self.metrics.get_gauge('last_batch_size'),
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate processing success rate."""
        processed = self.metrics.get_counter('dockerfiles_processed')
        errors = self.metrics.get_counter('processing_errors')
        total = processed + errors
        
        if total == 0:
            return 0.0
        
        return (processed / total) * 100.0