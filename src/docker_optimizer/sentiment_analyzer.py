"""
Sentiment Analysis Module for Docker Optimizer Agent

Provides sentiment-aware feedback for Dockerfile optimization recommendations,
enhancing user experience through empathetic and constructive communication.
"""

import re
import logging
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from functools import wraps, lru_cache
from contextlib import contextmanager
from collections import OrderedDict

try:
    from .i18n_sentiment import GlobalSentimentAnalyzer, SupportedLanguage, MultilingualFeedback
    GLOBAL_SUPPORT = True
except ImportError:
    GLOBAL_SUPPORT = False

class SentimentScore(Enum):
    """Sentiment scoring categories"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class FeedbackTone(Enum):
    """Feedback delivery tone categories"""
    ENCOURAGING = "encouraging"
    CONSTRUCTIVE = "constructive"
    INFORMATIVE = "informative"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class SentimentAnalysis:
    """Results of sentiment analysis on feedback text"""
    sentiment_score: SentimentScore
    confidence: float
    tone: FeedbackTone
    emotional_keywords: List[str]
    suggestions: List[str]
    processing_time_ms: float = 0.0
    text_length: int = 0
    error_details: Optional[str] = None

@dataclass 
class OptimizedFeedback:
    """Sentiment-optimized feedback for user delivery"""
    original_message: str
    optimized_message: str
    sentiment_analysis: SentimentAnalysis
    improvement_category: str
    optimization_applied: bool = True
    fallback_used: bool = False

class SentimentAnalyzerError(Exception):
    """Custom exception for sentiment analyzer errors"""
    pass

class ValidationError(SentimentAnalyzerError):
    """Validation error for input data"""
    pass

class ProcessingError(SentimentAnalyzerError):
    """Error during sentiment processing"""
    pass

class TTLCache:
    """
    High-performance Time-To-Live cache with thread safety for sentiment analysis results.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl_seconds
    
    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[any]:
        """Get value from cache if exists and not expired."""
        with self._lock:
            if key not in self._cache or self._is_expired(key):
                self._misses += 1
                return None
            
            # Move to end (LRU)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._hits += 1
            return value
    
    def put(self, key: str, value: any) -> None:
        """Put value in cache with TTL."""
        with self._lock:
            # Remove expired entries periodically
            if len(self._cache) % 100 == 0:
                self._evict_expired()
            
            # Evict oldest if at max size
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)
                self._timestamps.pop(oldest_key, None)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "cache_size": len(self._cache),
                "max_size": self.max_size
            }

class DockerfileSentimentAnalyzer:
    """
    Analyzes and optimizes sentiment in Dockerfile feedback messages.
    
    Transforms technical feedback into empathetic, constructive guidance
    that encourages best practices while maintaining technical accuracy.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, enable_metrics: bool = True,
                 enable_caching: bool = True, cache_size: int = 1000, cache_ttl_seconds: int = 3600,
                 max_workers: int = 4, enable_global: bool = True, target_region: str = "GLOBAL"):
        """
        Initialize DockerfileSentimentAnalyzer with robust error handling and performance optimization.
        
        Args:
            logger: Custom logger instance (creates default if None)
            enable_metrics: Enable performance and usage metrics collection
            enable_caching: Enable intelligent caching system
            cache_size: Maximum number of cached results
            cache_ttl_seconds: Cache entry time-to-live in seconds
            max_workers: Maximum threads for parallel processing
            enable_global: Enable global/multilingual support
            target_region: Target deployment region for compliance
        """
        self.logger = logger or self._setup_logger()
        self.enable_metrics = enable_metrics
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.enable_global = enable_global and GLOBAL_SUPPORT
        self.target_region = target_region
        
        # Global/multilingual support
        self.global_analyzer = GlobalSentimentAnalyzer() if self.enable_global else None
        
        # Performance optimization components
        self.cache = TTLCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds) if enable_caching else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sentiment_analyzer")
        
        # Enhanced metrics with performance tracking
        self.metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time_ms": 0.0,
            "sentiment_distribution": {},
            "error_counts": {},
            "cache_stats": {},
            "parallel_processing_count": 0,
            "performance_improvements": []
        }
        
        # Input validation parameters
        self.max_text_length = 10000  # Maximum characters to process
        self.min_text_length = 1      # Minimum characters to process
        
        # Sentiment keyword dictionaries
        self.positive_keywords = {
            'excellent', 'great', 'perfect', 'optimal', 'efficient', 
            'secure', 'clean', 'well-structured', 'best-practice',
            'recommended', 'improved', 'optimized', 'enhanced'
        }
        
        self.negative_keywords = {
            'error', 'failure', 'broken', 'vulnerable', 'insecure',
            'inefficient', 'bloated', 'deprecated', 'dangerous',
            'problematic', 'wrong', 'bad', 'poor', 'terrible'
        }
        
        self.warning_keywords = {
            'warning', 'caution', 'attention', 'consider', 'should',
            'might', 'potential', 'risk', 'issue', 'concern'
        }
        
        # Sentiment-aware response templates
        self.response_templates = {
            SentimentScore.VERY_POSITIVE: [
                "🎉 Excellent work! {}",
                "✨ Outstanding! {}",
                "🚀 Fantastic approach! {}"
            ],
            SentimentScore.POSITIVE: [
                "👍 Good job! {}",
                "✅ Nice work! {}",
                "🌟 Well done! {}"
            ],
            SentimentScore.NEUTRAL: [
                "ℹ️  {}",
                "📋 {}",
                "🔍 {}"
            ],
            SentimentScore.NEGATIVE: [
                "⚠️  Let's improve this: {}",
                "🔧 Here's how to fix this: {}",
                "💡 Consider this enhancement: {}"
            ],
            SentimentScore.VERY_NEGATIVE: [
                "🚨 Critical improvement needed: {}",
                "🛡️  Security enhancement required: {}",
                "⚡ Important optimization: {}"
            ]
        }
        
        # Initialize metrics
        if self.enable_metrics:
            self.logger.info(f"Sentiment analyzer initialized with metrics enabled, caching: {enable_caching}, max_workers: {max_workers}")

    def _generate_cache_key(self, text: str, operation: str = "sentiment") -> str:
        """Generate a unique cache key for sentiment analysis results."""
        # Create a hash of the text content and operation type
        content = f"{operation}:{text.strip().lower()}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=256)
    def _precompiled_patterns_cache(self) -> Dict[str, re.Pattern]:
        """Cache compiled regex patterns for better performance."""
        return {
            'suspicious_content': re.compile(r'<script|javascript:|data:text/html', re.IGNORECASE),
            'word_boundary': re.compile(r'\b\w+\b'),
            'whitespace': re.compile(r'\s+')
        }

    def _optimized_keyword_matching(self, text_lower: str, keyword_sets: List[set]) -> Tuple[List[int], List[str]]:
        """
        Optimized keyword matching using compiled patterns and efficient algorithms.
        
        Returns:
            Tuple of (counts_per_set, all_found_keywords)
        """
        # Use word boundary matching for better accuracy
        word_pattern = self._precompiled_patterns_cache()['word_boundary']
        words_in_text = set(word_pattern.findall(text_lower))
        
        counts = []
        found_keywords = []
        
        for keyword_set in keyword_sets:
            # Fast set intersection instead of loops
            matching_keywords = words_in_text.intersection(keyword_set)
            counts.append(len(matching_keywords))
            found_keywords.extend(list(matching_keywords))
        
        return counts, found_keywords

    def _setup_logger(self) -> logging.Logger:
        """Set up default logger for sentiment analysis."""
        logger = logging.getLogger("docker_optimizer.sentiment_analyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _validate_input(self, text: str, operation: str) -> None:
        """
        Validate input text for processing.
        
        Args:
            text: Input text to validate
            operation: Operation being performed (for error context)
            
        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValidationError(f"Input must be string, got {type(text).__name__}")
        
        if len(text) < self.min_text_length:
            raise ValidationError(f"Text too short (minimum {self.min_text_length} characters)")
            
        if len(text) > self.max_text_length:
            raise ValidationError(f"Text too long (maximum {self.max_text_length} characters)")
        
        # Check for potentially malicious content
        suspicious_patterns = [r'<script', r'javascript:', r'data:text/html']
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValidationError(f"Potentially malicious content detected in {operation}")

    def _update_metrics(self, operation: str, success: bool, processing_time_ms: float = 0.0,
                       sentiment: Optional[SentimentScore] = None, error_type: Optional[str] = None,
                       cache_hit: bool = False) -> None:
        """Update internal metrics for monitoring and performance analysis."""
        if not self.enable_metrics:
            return
            
        try:
            self.metrics["total_analyses"] += 1
            
            if success:
                self.metrics["successful_analyses"] += 1
                # Update average processing time
                current_avg = self.metrics["average_processing_time_ms"]
                total_successful = self.metrics["successful_analyses"]
                self.metrics["average_processing_time_ms"] = (
                    (current_avg * (total_successful - 1) + processing_time_ms) / total_successful
                )
                
                # Update sentiment distribution
                if sentiment:
                    sentiment_key = sentiment.value
                    self.metrics["sentiment_distribution"][sentiment_key] = (
                        self.metrics["sentiment_distribution"].get(sentiment_key, 0) + 1
                    )
            else:
                self.metrics["failed_analyses"] += 1
                if error_type:
                    self.metrics["error_counts"][error_type] = (
                        self.metrics["error_counts"].get(error_type, 0) + 1
                    )
            
            # Update cache statistics
            if self.cache and self.enable_caching:
                try:
                    self.metrics["cache_stats"] = self.cache.get_stats()
                except Exception as cache_error:
                    self.logger.warning(f"Failed to update cache stats: {cache_error}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {e}")

    @contextmanager
    def _performance_monitoring(self, operation: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        success = False
        error_type = None
        sentiment = None
        
        try:
            yield
            success = True
        except ValidationError as e:
            error_type = "ValidationError"
            self.logger.warning(f"Validation error in {operation}: {e}")
            raise
        except ProcessingError as e:
            error_type = "ProcessingError"
            self.logger.error(f"Processing error in {operation}: {e}")
            raise
        except Exception as e:
            error_type = "UnknownError"
            self.logger.error(f"Unexpected error in {operation}: {e}")
            raise ProcessingError(f"Unexpected error during {operation}: {str(e)}")
        finally:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(operation, success, processing_time_ms, sentiment, error_type)

    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of feedback text with comprehensive error handling and caching.
        
        Args:
            text: Original feedback message
            
        Returns:
            SentimentAnalysis with score, confidence, and recommendations
            
        Raises:
            ValidationError: If input text is invalid
            ProcessingError: If sentiment analysis fails
        """
        start_time = time.time()
        processing_time_ms = 0.0
        error_details = None
        cache_hit = False
        
        try:
            # Input validation
            self._validate_input(text, "sentiment_analysis")
            
            # Check cache first if enabled
            cache_key = None
            if self.cache and self.enable_caching:
                cache_key = self._generate_cache_key(text, "sentiment_analysis")
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    cache_hit = True
                    processing_time_ms = (time.time() - start_time) * 1000
                    self._update_metrics("sentiment_analysis", True, processing_time_ms, 
                                       cached_result.sentiment_score, cache_hit=True)
                    self.logger.debug("Cache hit for sentiment analysis")
                    return cached_result
            
            # Safe text processing
            with self._performance_monitoring("sentiment_analysis"):
                text_lower = text.lower().strip()
                text_length = len(text)
                
                if not text_lower:  # Empty after stripping
                    self.logger.warning("Empty text provided for sentiment analysis")
                    return self._create_fallback_analysis(text, "Empty input text")
                
                # Optimized sentiment indicators counting
                try:
                    keyword_sets = [self.positive_keywords, self.negative_keywords, self.warning_keywords]
                    counts, emotional_keywords = self._optimized_keyword_matching(text_lower, keyword_sets)
                    positive_count, negative_count, warning_count = counts
                except Exception as e:
                    # Fallback to slower method if optimization fails
                    self.logger.warning(f"Optimized keyword matching failed, using fallback: {e}")
                    try:
                        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
                        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
                        warning_count = sum(1 for word in self.warning_keywords if word in text_lower)
                        
                        emotional_keywords = []
                        emotional_keywords.extend([word for word in self.positive_keywords if word in text_lower])
                        emotional_keywords.extend([word for word in self.negative_keywords if word in text_lower])
                        emotional_keywords.extend([word for word in self.warning_keywords if word in text_lower])
                    except Exception as fallback_error:
                        raise ProcessingError(f"Failed to count sentiment keywords: {str(fallback_error)}")
                
                # Determine sentiment score with fallback logic
                try:
                    total_sentiment = positive_count - negative_count
                    
                    if total_sentiment >= 3:
                        sentiment = SentimentScore.VERY_POSITIVE
                        tone = FeedbackTone.ENCOURAGING
                    elif total_sentiment >= 1:
                        sentiment = SentimentScore.POSITIVE
                        tone = FeedbackTone.CONSTRUCTIVE
                    elif total_sentiment <= -3:
                        sentiment = SentimentScore.VERY_NEGATIVE
                        tone = FeedbackTone.CRITICAL
                    elif total_sentiment <= -1:
                        sentiment = SentimentScore.NEGATIVE
                        tone = FeedbackTone.WARNING
                    else:
                        sentiment = SentimentScore.NEUTRAL
                        tone = FeedbackTone.INFORMATIVE
                except Exception as e:
                    self.logger.warning(f"Failed to determine sentiment score: {e}")
                    sentiment = SentimentScore.NEUTRAL
                    tone = FeedbackTone.INFORMATIVE
                    
                # Calculate confidence with bounds checking
                try:
                    total_words = len(text.split())
                    if total_words == 0:
                        confidence = 0.5  # Base confidence for empty content
                    else:
                        keyword_density = len(emotional_keywords) / total_words
                        confidence = min(0.9, max(0.3, 0.5 + keyword_density))  # Bounded confidence
                except Exception as e:
                    self.logger.warning(f"Failed to calculate confidence: {e}")
                    confidence = 0.5  # Fallback confidence
                
                # Generate improvement suggestions with error handling
                try:
                    suggestions = self._generate_suggestions(sentiment, emotional_keywords)
                except Exception as e:
                    self.logger.warning(f"Failed to generate suggestions: {e}")
                    suggestions = ["Review feedback tone and clarity"]
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Create result object
                result = SentimentAnalysis(
                    sentiment_score=sentiment,
                    confidence=confidence,
                    tone=tone,
                    emotional_keywords=emotional_keywords,
                    suggestions=suggestions,
                    processing_time_ms=processing_time_ms,
                    text_length=text_length,
                    error_details=error_details
                )
                
                # Cache the result if caching is enabled
                if self.cache and self.enable_caching and cache_key:
                    try:
                        self.cache.put(cache_key, result)
                        self.logger.debug("Cached sentiment analysis result")
                    except Exception as cache_error:
                        self.logger.warning(f"Failed to cache result: {cache_error}")
                
                # Update metrics with successful analysis
                self._update_metrics("sentiment_analysis", True, processing_time_ms, sentiment, cache_hit=cache_hit)
                
                return result
                
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except ProcessingError:
            # Re-raise processing errors as-is
            raise
        except Exception as e:
            # Handle unexpected errors with fallback
            error_details = str(e)
            self.logger.error(f"Unexpected error in sentiment analysis: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics("sentiment_analysis", False, processing_time_ms, error_type="UnexpectedError")
            
            # Return fallback analysis instead of failing completely
            return self._create_fallback_analysis(text, error_details)

    def _create_fallback_analysis(self, text: str, error_details: str) -> SentimentAnalysis:
        """Create a safe fallback sentiment analysis when primary analysis fails."""
        return SentimentAnalysis(
            sentiment_score=SentimentScore.NEUTRAL,
            confidence=0.3,
            tone=FeedbackTone.INFORMATIVE,
            emotional_keywords=[],
            suggestions=["Unable to perform detailed sentiment analysis"],
            processing_time_ms=0.0,
            text_length=len(text),
            error_details=error_details
        )

    def optimize_feedback(self, 
                         original_message: str, 
                         category: str = "optimization") -> OptimizedFeedback:
        """
        Transform technical feedback into sentiment-optimized messaging with robust error handling.
        
        Args:
            original_message: Original technical feedback
            category: Type of optimization (security, performance, etc.)
            
        Returns:
            OptimizedFeedback with enhanced user experience
            
        Raises:
            ValidationError: If input is invalid
            ProcessingError: If feedback optimization fails
        """
        fallback_used = False
        optimization_applied = True
        
        try:
            # Validate inputs
            self._validate_input(original_message, "feedback_optimization")
            if not isinstance(category, str) or not category.strip():
                self.logger.warning("Invalid category provided, using default")
                category = "optimization"
            
            # Perform sentiment analysis with error handling
            try:
                sentiment_analysis = self.analyze_sentiment(original_message)
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed during feedback optimization: {e}")
                # Create fallback sentiment analysis
                sentiment_analysis = self._create_fallback_analysis(original_message, str(e))
                fallback_used = True
            
            # Select appropriate template based on sentiment with fallbacks
            try:
                templates = self.response_templates.get(sentiment_analysis.sentiment_score)
                if not templates:
                    self.logger.warning(f"No templates found for sentiment: {sentiment_analysis.sentiment_score}")
                    templates = self.response_templates[SentimentScore.NEUTRAL]
                    fallback_used = True
                
                template = templates[0]  # Use first template for consistency
            except (KeyError, IndexError, TypeError) as e:
                self.logger.warning(f"Template selection failed: {e}")
                template = "ℹ️ {}"  # Safe fallback template
                fallback_used = True
            
            # Clean and enhance the original message with error handling
            try:
                enhanced_message = self._enhance_message(original_message, sentiment_analysis)
            except Exception as e:
                self.logger.warning(f"Message enhancement failed: {e}")
                enhanced_message = original_message  # Use original if enhancement fails
                optimization_applied = False
                fallback_used = True
            
            # Format final optimized message with safe formatting
            try:
                optimized_message = template.format(enhanced_message)
            except (ValueError, TypeError, KeyError) as e:
                self.logger.warning(f"Template formatting failed: {e}")
                optimized_message = f"ℹ️ {enhanced_message}"  # Safe fallback formatting
                fallback_used = True
            
            # Validate output isn't empty or malformed
            if not optimized_message or len(optimized_message.strip()) < 2:
                self.logger.warning("Optimized message is empty or too short, using fallback")
                optimized_message = f"📋 {original_message}"
                fallback_used = True
                optimization_applied = False
            
            # Log successful optimization
            if not fallback_used:
                self.logger.debug(f"Successfully optimized feedback for category: {category}")
            
            return OptimizedFeedback(
                original_message=original_message,
                optimized_message=optimized_message,
                sentiment_analysis=sentiment_analysis,
                improvement_category=category,
                optimization_applied=optimization_applied,
                fallback_used=fallback_used
            )
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Handle unexpected errors with complete fallback
            self.logger.error(f"Complete failure in feedback optimization: {e}")
            fallback_analysis = self._create_fallback_analysis(original_message, str(e))
            
            return OptimizedFeedback(
                original_message=original_message,
                optimized_message=f"📋 {original_message}",  # Safe fallback
                sentiment_analysis=fallback_analysis,
                improvement_category=category,
                optimization_applied=False,
                fallback_used=True
            )

    def _enhance_message(self, message: str, analysis: SentimentAnalysis) -> str:
        """Enhance message content based on sentiment analysis."""
        enhanced = message
        
        # Replace harsh technical terms with friendlier alternatives
        replacements = {
            r'\berror\b': 'issue',
            r'\bfailed\b': 'needs attention',
            r'\bbad\b': 'suboptimal',
            r'\bwrong\b': 'could be improved',
            r'\bfix\b': 'enhance',
            r'\bbroken\b': 'needs updating'
        }
        
        for pattern, replacement in replacements.items():
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
            
        return enhanced

    def _generate_suggestions(self, 
                            sentiment: SentimentScore, 
                            keywords: List[str]) -> List[str]:
        """Generate contextual improvement suggestions."""
        suggestions = []
        
        if sentiment in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE]:
            suggestions.extend([
                "Consider breaking down the feedback into smaller, actionable steps",
                "Add positive reinforcement for existing good practices",
                "Provide clear examples of the recommended solution"
            ])
        elif sentiment == SentimentScore.NEUTRAL:
            suggestions.extend([
                "Add encouraging language to motivate improvement",
                "Include benefits of implementing the suggestion"
            ])
            
        if 'security' in ' '.join(keywords):
            suggestions.append("Frame security improvements as protecting user data")
        if 'performance' in ' '.join(keywords):
            suggestions.append("Highlight efficiency gains and cost savings")
            
        return suggestions

    def batch_analyze_feedback(self, feedback_list: List[str]) -> Dict[str, OptimizedFeedback]:
        """Analyze and optimize multiple feedback messages."""
        results = {}
        
        for i, feedback in enumerate(feedback_list):
            optimized = self.optimize_feedback(feedback, f"optimization_{i}")
            results[f"feedback_{i}"] = optimized
            
        return results

    def generate_sentiment_report(self, feedback_list: List[str]) -> Dict:
        """Generate comprehensive sentiment analysis report."""
        analyses = [self.analyze_sentiment(fb) for fb in feedback_list]
        
        sentiment_distribution = {}
        for analysis in analyses:
            sentiment = analysis.sentiment_score.value
            sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses)
        
        all_keywords = []
        for analysis in analyses:
            all_keywords.extend(analysis.emotional_keywords)
        
        keyword_frequency = {}
        for keyword in all_keywords:
            keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
            
        return {
            "total_feedback_analyzed": len(feedback_list),
            "sentiment_distribution": sentiment_distribution,
            "average_confidence": round(avg_confidence, 3),
            "top_emotional_keywords": sorted(keyword_frequency.items(), 
                                           key=lambda x: x[1], reverse=True)[:10],
            "recommendations": self._generate_overall_recommendations(analyses)
        }

    def _generate_overall_recommendations(self, analyses: List[SentimentAnalysis]) -> List[str]:
        """Generate system-wide recommendations based on sentiment patterns."""
        recommendations = []
        
        negative_ratio = sum(1 for a in analyses 
                           if a.sentiment_score in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE]) / len(analyses)
        
        if negative_ratio > 0.5:
            recommendations.append("Consider implementing more positive reinforcement in feedback")
        
        if negative_ratio > 0.3:
            recommendations.append("Review feedback templates for empathetic language")
            
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses)
        if avg_confidence < 0.6:
            recommendations.append("Enhance sentiment detection with more comprehensive keyword dictionaries")
            
        return recommendations

    def get_health_status(self) -> Dict:
        """
        Get health status and performance metrics for monitoring.
        
        Returns:
            Dictionary with health metrics, performance data, and system status
        """
        try:
            total_analyses = self.metrics.get("total_analyses", 0)
            successful_analyses = self.metrics.get("successful_analyses", 0)
            failed_analyses = self.metrics.get("failed_analyses", 0)
            
            # Calculate success rate
            success_rate = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0.0
            
            # Determine health status
            if success_rate >= 95:
                health_status = "HEALTHY"
            elif success_rate >= 80:
                health_status = "DEGRADED"
            else:
                health_status = "UNHEALTHY"
            
            return {
                "status": health_status,
                "success_rate_percent": round(success_rate, 2),
                "total_analyses": total_analyses,
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "average_processing_time_ms": round(self.metrics.get("average_processing_time_ms", 0.0), 2),
                "sentiment_distribution": self.metrics.get("sentiment_distribution", {}),
                "error_counts": self.metrics.get("error_counts", {}),
                "metrics_enabled": self.enable_metrics,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate health status: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": time.time()
            }

    def reset_metrics(self) -> None:
        """Reset all internal metrics (useful for testing or periodic cleanup)."""
        try:
            self.metrics = {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "average_processing_time_ms": 0.0,
                "sentiment_distribution": {},
                "error_counts": {}
            }
            self.logger.info("Metrics successfully reset")
        except Exception as e:
            self.logger.error(f"Failed to reset metrics: {e}")

    def batch_analyze_feedback_robust(self, feedback_list: List[str], 
                                    max_failures: int = 5) -> Tuple[Dict[str, OptimizedFeedback], List[str]]:
        """
        Robust batch analysis with failure tolerance.
        
        Args:
            feedback_list: List of feedback messages to analyze
            max_failures: Maximum number of failures before aborting batch
            
        Returns:
            Tuple of (successful_results, failed_items)
        """
        if not isinstance(feedback_list, list):
            raise ValidationError("feedback_list must be a list")
        
        results = {}
        failures = []
        failure_count = 0
        
        for i, feedback in enumerate(feedback_list):
            try:
                if failure_count >= max_failures:
                    self.logger.warning(f"Aborting batch analysis after {failure_count} failures")
                    failures.extend(feedback_list[i:])  # Add remaining items as failures
                    break
                
                optimized = self.optimize_feedback(feedback, f"batch_optimization_{i}")
                results[f"feedback_{i}"] = optimized
                
            except Exception as e:
                failure_count += 1
                error_msg = f"Failed to process feedback {i}: {str(e)}"
                failures.append(error_msg)
                self.logger.warning(error_msg)
                
        self.logger.info(f"Batch analysis completed: {len(results)} successful, {len(failures)} failed")
        return results, failures

    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate sentiment analyzer configuration and setup.
        
        Returns:
            Dictionary with validation results for different components
        """
        validation_results = {}
        
        try:
            # Check keyword dictionaries
            validation_results["positive_keywords_loaded"] = (
                isinstance(self.positive_keywords, set) and len(self.positive_keywords) > 0
            )
            validation_results["negative_keywords_loaded"] = (
                isinstance(self.negative_keywords, set) and len(self.negative_keywords) > 0
            )
            validation_results["warning_keywords_loaded"] = (
                isinstance(self.warning_keywords, set) and len(self.warning_keywords) > 0
            )
            
            # Check response templates
            validation_results["response_templates_loaded"] = (
                isinstance(self.response_templates, dict) and 
                len(self.response_templates) == len(SentimentScore)
            )
            
            # Check logger setup
            validation_results["logger_configured"] = (
                self.logger is not None and hasattr(self.logger, 'info')
            )
            
            # Check metrics system
            validation_results["metrics_system_active"] = (
                isinstance(self.metrics, dict) and self.enable_metrics
            )
            
            # Check validation parameters
            validation_results["validation_params_set"] = (
                isinstance(self.max_text_length, int) and self.max_text_length > 0 and
                isinstance(self.min_text_length, int) and self.min_text_length >= 0
            )
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            validation_results["validation_error"] = str(e)
            
        return validation_results

    def parallel_analyze_batch(self, feedback_list: List[str], 
                             max_workers: Optional[int] = None) -> Dict[str, OptimizedFeedback]:
        """
        High-performance parallel batch processing of feedback messages.
        
        Args:
            feedback_list: List of feedback messages to process
            max_workers: Override default max_workers for this batch
            
        Returns:
            Dictionary of optimized feedback results
        """
        if not feedback_list:
            return {}
            
        if not isinstance(feedback_list, list):
            raise ValidationError("feedback_list must be a list")
        
        workers = max_workers or self.max_workers
        batch_start_time = time.time()
        
        def process_single_feedback(indexed_feedback: Tuple[int, str]) -> Tuple[int, OptimizedFeedback]:
            """Process a single feedback message with error handling."""
            index, feedback = indexed_feedback
            try:
                result = self.optimize_feedback(feedback, f"parallel_batch_{index}")
                return index, result
            except Exception as e:
                self.logger.warning(f"Failed to process feedback {index}: {e}")
                # Return fallback result
                fallback_analysis = self._create_fallback_analysis(feedback, str(e))
                fallback_feedback = OptimizedFeedback(
                    original_message=feedback,
                    optimized_message=f"📋 {feedback}",
                    sentiment_analysis=fallback_analysis,
                    improvement_category=f"parallel_batch_{index}",
                    optimization_applied=False,
                    fallback_used=True
                )
                return index, fallback_feedback
        
        results = {}
        
        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sentiment_parallel") as executor:
                # Submit all tasks
                indexed_feedback = list(enumerate(feedback_list))
                future_to_index = {
                    executor.submit(process_single_feedback, item): item[0] 
                    for item in indexed_feedback
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    try:
                        index, result = future.result(timeout=30)  # 30 second timeout per task
                        results[f"feedback_{index}"] = result
                    except Exception as e:
                        index = future_to_index[future]
                        self.logger.error(f"Parallel processing failed for feedback {index}: {e}")
                        # Add fallback result
                        fallback_feedback = OptimizedFeedback(
                            original_message=feedback_list[index] if index < len(feedback_list) else "Unknown",
                            optimized_message="📋 Processing failed",
                            sentiment_analysis=self._create_fallback_analysis("", str(e)),
                            improvement_category=f"parallel_batch_{index}",
                            optimization_applied=False,
                            fallback_used=True
                        )
                        results[f"feedback_{index}"] = fallback_feedback
                        
        except Exception as e:
            self.logger.error(f"Parallel batch processing failed: {e}")
            # Fallback to sequential processing
            self.logger.info("Falling back to sequential processing")
            return self.batch_analyze_feedback(feedback_list)
        
        batch_time_ms = (time.time() - batch_start_time) * 1000
        self.logger.info(f"Parallel batch processing completed: {len(results)} items in {batch_time_ms:.2f}ms")
        
        # Update metrics
        if self.enable_metrics:
            self.metrics["parallel_processing_count"] += 1
            self.metrics["performance_improvements"].append({
                "type": "parallel_batch",
                "items_processed": len(results),
                "processing_time_ms": batch_time_ms,
                "workers_used": workers,
                "timestamp": time.time()
            })
        
        return results

    def get_performance_report(self) -> Dict:
        """
        Generate comprehensive performance analysis report.
        
        Returns:
            Dictionary with detailed performance metrics and recommendations
        """
        try:
            health_status = self.get_health_status()
            cache_stats = self.cache.get_stats() if self.cache else {"enabled": False}
            
            # Calculate performance insights
            total_analyses = self.metrics.get("total_analyses", 0)
            avg_time = self.metrics.get("average_processing_time_ms", 0.0)
            parallel_count = self.metrics.get("parallel_processing_count", 0)
            
            # Performance classification
            if avg_time < 10:
                performance_grade = "A"
                performance_desc = "Excellent"
            elif avg_time < 50:
                performance_grade = "B"
                performance_desc = "Good"
            elif avg_time < 100:
                performance_grade = "C"
                performance_desc = "Fair"
            else:
                performance_grade = "D"
                performance_desc = "Needs Improvement"
            
            # Generate recommendations
            recommendations = []
            if cache_stats.get("hit_rate_percent", 0) < 50:
                recommendations.append("Consider increasing cache size or TTL for better performance")
            if parallel_count == 0 and total_analyses > 10:
                recommendations.append("Use parallel processing for batch operations")
            if avg_time > 100:
                recommendations.append("Consider optimizing keyword dictionaries or reducing text processing")
            
            return {
                "performance_grade": performance_grade,
                "performance_description": performance_desc,
                "average_processing_time_ms": round(avg_time, 2),
                "total_analyses": total_analyses,
                "parallel_processing_usage": parallel_count,
                "cache_performance": cache_stats,
                "health_status": health_status["status"],
                "success_rate_percent": health_status.get("success_rate_percent", 0.0),
                "recommendations": recommendations,
                "performance_improvements": self.metrics.get("performance_improvements", [])[-5:],  # Last 5 improvements
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {
                "error": f"Performance report generation failed: {str(e)}",
                "timestamp": time.time()
            }

    def clear_cache(self) -> bool:
        """Clear the sentiment analysis cache."""
        try:
            if self.cache:
                self.cache.clear()
                self.logger.info("Sentiment analysis cache cleared")
                return True
            else:
                self.logger.info("Cache not enabled")
                return False
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    def get_multilingual_feedback(self, 
                                 original_message: str,
                                 sentiment_category: str = "neutral",
                                 target_languages: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Generate multilingual feedback for global deployment.
        
        Args:
            original_message: Original feedback message
            sentiment_category: Sentiment category
            target_languages: Target language codes (e.g., ['es', 'fr', 'de'])
            
        Returns:
            Multilingual feedback dictionary or None if global support disabled
        """
        if not self.enable_global or not self.global_analyzer:
            self.logger.warning("Global multilingual support not enabled")
            return None
        
        try:
            # Convert language codes to SupportedLanguage enums
            if target_languages:
                supported_languages = []
                for lang_code in target_languages:
                    try:
                        supported_languages.append(SupportedLanguage(lang_code))
                    except ValueError:
                        self.logger.warning(f"Unsupported language code: {lang_code}")
            else:
                supported_languages = None
            
            multilingual_feedback = self.global_analyzer.get_multilingual_feedback(
                original_message, sentiment_category, supported_languages
            )
            
            return {
                "english": multilingual_feedback.english,
                "translations": multilingual_feedback.translations,
                "detected_language": multilingual_feedback.detected_language,
                "confidence": multilingual_feedback.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate multilingual feedback: {e}")
            return None

    def get_global_deployment_report(self, 
                                   feedback_messages: List[str]) -> Dict:
        """
        Generate a comprehensive global deployment readiness report.
        
        Args:
            feedback_messages: Sample feedback messages to analyze
            
        Returns:
            Global deployment report with recommendations
        """
        if not self.enable_global or not self.global_analyzer:
            return {
                "global_support": "disabled",
                "recommendation": "Enable global support for international deployment"
            }
        
        try:
            report = self.global_analyzer.generate_global_report(
                feedback_messages, self.target_region
            )
            
            # Add analyzer-specific metrics
            health_status = self.get_health_status()
            performance_report = self.get_performance_report()
            
            report["analyzer_metrics"] = {
                "health_status": health_status["status"],
                "success_rate": health_status.get("success_rate_percent", 0),
                "performance_grade": performance_report.get("performance_grade", "Unknown"),
                "cache_enabled": self.enable_caching,
                "parallel_processing": self.max_workers > 1
            }
            
            # Add global-specific recommendations
            if report["global_readiness"]["score"] < 90:
                report["deployment_recommendations"].extend([
                    "Enable performance caching for global deployment",
                    "Use parallel processing for high-volume international usage",
                    "Monitor health metrics across different regions"
                ])
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate global deployment report: {e}")
            return {
                "error": f"Global report generation failed: {str(e)}",
                "global_support": "enabled_but_failed"
            }

    def apply_cultural_adaptation(self, 
                                feedback: str, 
                                target_language: str = "en") -> str:
        """
        Apply cultural adaptation to feedback based on target language.
        
        Args:
            feedback: Original feedback message
            target_language: Target language code for cultural adaptation
            
        Returns:
            Culturally adapted feedback
        """
        if not self.enable_global or not self.global_analyzer:
            return feedback
        
        try:
            # Convert language code to SupportedLanguage enum
            target_lang_enum = SupportedLanguage(target_language)
            adapted_feedback = self.global_analyzer.apply_cultural_sensitivity(
                feedback, target_lang_enum
            )
            return adapted_feedback
            
        except (ValueError, Exception) as e:
            self.logger.warning(f"Cultural adaptation failed for {target_language}: {e}")
            return feedback

    def get_regional_compliance_info(self) -> Dict:
        """
        Get regional compliance information for current target region.
        
        Returns:
            Compliance information dictionary
        """
        if not self.enable_global or not self.global_analyzer:
            return {"compliance": "not_available", "region": self.target_region}
        
        try:
            compliance_notice = self.global_analyzer.get_regional_compliance_notice(self.target_region)
            timezone_greeting = self.global_analyzer.get_timezone_aware_greeting()
            
            return {
                "target_region": self.target_region,
                "compliance_notice": compliance_notice,
                "timezone_greeting": timezone_greeting,
                "supported_languages": [lang.value for lang in SupportedLanguage],
                "cultural_sensitivity": "enabled"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get compliance info: {e}")
            return {"error": str(e), "region": self.target_region}

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors