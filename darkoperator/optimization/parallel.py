"""Parallel processing utilities for scaling physics computations."""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Iterator, Optional, Union, Tuple
import numpy as np
import logging
import time
from queue import Queue
import threading


logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Parallel processor for CPU-intensive physics computations."""
    
    def __init__(self, n_workers: Optional[int] = None, use_processes: bool = True):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_processes = use_processes
        
        # Choose executor type
        if use_processes:
            self.executor_class = ProcessPoolExecutor
            logger.info(f"Using {self.n_workers} processes for parallel computation")
        else:
            self.executor_class = ThreadPoolExecutor
            logger.info(f"Using {self.n_workers} threads for parallel computation")
    
    def map(self, func: Callable, iterable: List[Any], chunk_size: Optional[int] = None) -> List[Any]:
        """
        Parallel map operation.
        
        Args:
            func: Function to apply
            iterable: Items to process
            chunk_size: Chunk size for batching
            
        Returns:
            List of results
        """
        if chunk_size is None:
            chunk_size = max(1, len(iterable) // (self.n_workers * 4))
        
        results = []
        
        with self.executor_class(max_workers=self.n_workers) as executor:
            # Submit chunks
            futures = []
            for i in range(0, len(iterable), chunk_size):
                chunk = iterable[i:i + chunk_size]
                future = executor.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Parallel processing error: {e}")
                    raise
        
        return results
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def map_async(self, func: Callable, iterable: List[Any]) -> Iterator[Any]:
        """
        Asynchronous parallel map that yields results as they complete.
        
        Args:
            func: Function to apply
            iterable: Items to process
            
        Yields:
            Results as they complete
        """
        with self.executor_class(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(func, item) for item in iterable]
            
            # Yield results as they complete
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"Async processing error: {e}")
                    yield None


class BatchProcessor:
    """Efficient batch processing for neural operators."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 32,
        device: str = "auto",
        mixed_precision: bool = True
    ):
        self.model = model
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup mixed precision if available
        self.scaler = None
        if mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
    
    def process_dataset(
        self,
        dataset: Union[Dataset, List[torch.Tensor]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[torch.Tensor]:
        """
        Process entire dataset in batches.
        
        Args:
            dataset: Dataset or list of tensors to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed results
        """
        # Create dataloader if needed
        if isinstance(dataset, list):
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )
        
        results = []
        total_batches = len(dataloader)
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
                else:
                    batch = batch.to(self.device)
                
                # Process batch with optional mixed precision
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = self.model(batch)
                else:
                    output = self.model(batch)
                
                # Move results back to CPU to save GPU memory
                if torch.is_tensor(output):
                    output = output.cpu()
                elif isinstance(output, (list, tuple)):
                    output = [o.cpu() if torch.is_tensor(o) else o for o in output]
                
                results.append(output)
                
                # Progress callback
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
        
        return results
    
    def process_streaming(
        self,
        data_stream: Iterator[torch.Tensor],
        max_queue_size: int = 10
    ) -> Iterator[torch.Tensor]:
        """
        Process streaming data with efficient batching.
        
        Args:
            data_stream: Iterator yielding data tensors
            max_queue_size: Maximum queue size for buffering
            
        Yields:
            Processed results
        """
        batch_queue = Queue(maxsize=max_queue_size)
        result_queue = Queue(maxsize=max_queue_size)
        
        def batch_collector():
            """Collect items into batches."""
            batch = []
            for item in data_stream:
                batch.append(item)
                if len(batch) >= self.batch_size:
                    batch_queue.put(torch.stack(batch))
                    batch = []
            
            # Handle remaining items
            if batch:
                batch_queue.put(torch.stack(batch))
            
            batch_queue.put(None)  # Signal end
        
        def batch_processor():
            """Process batches."""
            self.model.eval()
            with torch.no_grad():
                while True:
                    batch = batch_queue.get()
                    if batch is None:
                        break
                    
                    batch = batch.to(self.device)
                    
                    if self.mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            output = self.model(batch)
                    else:
                        output = self.model(batch)
                    
                    result_queue.put(output.cpu())
            
            result_queue.put(None)  # Signal end
        
        # Start threads
        collector_thread = threading.Thread(target=batch_collector)
        processor_thread = threading.Thread(target=batch_processor)
        
        collector_thread.start()
        processor_thread.start()
        
        # Yield results
        while True:
            result = result_queue.get()
            if result is None:
                break
            
            # Unbatch results
            for i in range(result.shape[0]):
                yield result[i]
        
        # Wait for threads to complete
        collector_thread.join()
        processor_thread.join()


class DistributedProcessor:
    """Distributed processing across multiple GPUs/nodes."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(model)
            logger.info(f"Using {self.device_count} GPUs for distributed processing")
    
    def process_distributed(
        self,
        dataloader: DataLoader,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[torch.Tensor]:
        """
        Process data using multiple GPUs.
        
        Args:
            dataloader: DataLoader for input data
            progress_callback: Optional progress callback
            
        Returns:
            List of processed results
        """
        results = []
        total_batches = len(dataloader)
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                start_time = time.time()
                
                # Move to GPU(s)
                if torch.cuda.is_available():
                    batch = batch.cuda()
                
                # Process on multiple GPUs
                output = self.model(batch)
                
                # Move back to CPU
                output = output.cpu()
                results.append(output)
                
                # Log performance
                batch_time = time.time() - start_time
                if batch_idx % 10 == 0:
                    items_per_sec = batch.shape[0] / batch_time
                    logger.debug(f"Batch {batch_idx}/{total_batches}: {items_per_sec:.1f} items/sec")
                
                # Progress callback
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
        
        return results


class AsyncEventProcessor:
    """Asynchronous event processing for real-time applications."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 64,
        max_latency_ms: float = 100.0,
        device: str = "auto"
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_latency_s = max_latency_ms / 1000.0
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Batching state
        self.pending_events = []
        self.pending_futures = []
        self.last_batch_time = time.time()
        self.processing_lock = threading.Lock()
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
    
    def process_event_async(self, event: torch.Tensor) -> 'Future':
        """
        Process single event asynchronously.
        
        Args:
            event: Event tensor to process
            
        Returns:
            Future that will contain the result
        """
        from concurrent.futures import Future
        
        future = Future()
        
        with self.processing_lock:
            self.pending_events.append(event)
            self.pending_futures.append(future)
            
            # Trigger batch processing if needed
            should_process = (
                len(self.pending_events) >= self.max_batch_size or
                time.time() - self.last_batch_time > self.max_latency_s
            )
            
            if should_process:
                self._process_pending_batch()
        
        return future
    
    def _processing_loop(self):
        """Main processing loop for batched inference."""
        while self.running:
            time.sleep(0.01)  # 10ms check interval
            
            with self.processing_lock:
                if (self.pending_events and 
                    time.time() - self.last_batch_time > self.max_latency_s):
                    self._process_pending_batch()
    
    def _process_pending_batch(self):
        """Process current batch of pending events."""
        if not self.pending_events:
            return
        
        events = self.pending_events[:]
        futures = self.pending_futures[:]
        
        # Clear pending lists
        self.pending_events.clear()
        self.pending_futures.clear()
        self.last_batch_time = time.time()
        
        try:
            # Stack events into batch
            batch = torch.stack(events).to(self.device)
            
            # Process batch
            with torch.no_grad():
                results = self.model(batch)
            
            # Set results on futures
            results = results.cpu()
            for i, future in enumerate(futures):
                future.set_result(results[i])
                
        except Exception as e:
            # Set exception on all futures
            for future in futures:
                future.set_exception(e)
    
    def shutdown(self):
        """Shutdown async processor."""
        self.running = False
        self.processing_thread.join()
        
        # Process any remaining events
        with self.processing_lock:
            if self.pending_events:
                self._process_pending_batch()