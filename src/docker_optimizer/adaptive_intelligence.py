"""Adaptive Intelligence Engine for Self-Learning Docker Optimization."""

import json
import logging
import pickle
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .models import OptimizationResult

logger = logging.getLogger(__name__)


class OptimizationPattern(BaseModel):
    """Represents a learned optimization pattern."""
    
    pattern_id: str
    dockerfile_features: Dict[str, Any]
    optimization_strategy: str
    success_metrics: Dict[str, float]
    usage_count: int = 0
    last_used: datetime = Field(default_factory=datetime.now)
    confidence_score: float = 0.0


class LearningContext(BaseModel):
    """Context for learning from optimization results."""
    
    dockerfile_type: str
    project_language: str
    optimization_goals: List[str]
    performance_metrics: Dict[str, float]
    user_feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AdaptiveOptimizationEngine:
    """Self-learning optimization engine that adapts to user patterns."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the adaptive optimization engine."""
        self.cache_dir = cache_dir or Path.home() / ".docker_optimizer" / "adaptive_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning models
        self.pattern_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_extractor = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Pattern storage
        self.learned_patterns: Dict[str, OptimizationPattern] = {}
        self.learning_history: List[LearningContext] = []
        
        # Performance tracking
        self.optimization_metrics: Dict[str, List[float]] = defaultdict(list)
        self.user_preferences: Dict[str, Any] = {}
        
        # Load existing patterns
        self._load_learned_patterns()
        
        logger.info("Adaptive optimization engine initialized")
    
    def analyze_dockerfile_context(self, dockerfile_content: str) -> Dict[str, Any]:
        """Extract contextual features from Dockerfile."""
        features = {
            'base_image_type': self._extract_base_image_type(dockerfile_content),
            'language_indicators': self._detect_language_patterns(dockerfile_content),
            'complexity_score': self._calculate_complexity_score(dockerfile_content),
            'security_patterns': self._identify_security_patterns(dockerfile_content),
            'performance_indicators': self._extract_performance_indicators(dockerfile_content),
            'instruction_count': len(dockerfile_content.split('\n')),
            'layer_count': dockerfile_content.count('RUN') + dockerfile_content.count('COPY') + dockerfile_content.count('ADD'),
            'has_multistage': 'AS ' in dockerfile_content.upper(),
            'has_healthcheck': 'HEALTHCHECK' in dockerfile_content.upper(),
            'has_user': 'USER ' in dockerfile_content.upper(),
            'package_managers': self._detect_package_managers(dockerfile_content)
        }
        
        return features
    
    def suggest_adaptive_optimization(self, dockerfile_content: str, user_goals: List[str] = None) -> Dict[str, Any]:
        """Suggest optimizations based on learned patterns."""
        user_goals = user_goals or ['security', 'size', 'performance']
        
        # Extract features
        features = self.analyze_dockerfile_context(dockerfile_content)
        
        # Find similar patterns
        similar_patterns = self._find_similar_patterns(features)
        
        # Generate adaptive suggestions
        suggestions = {
            'primary_strategy': self._select_primary_strategy(features, similar_patterns, user_goals),
            'adaptive_optimizations': self._generate_adaptive_optimizations(features, similar_patterns),
            'confidence_score': self._calculate_suggestion_confidence(similar_patterns),
            'learning_insights': self._extract_learning_insights(similar_patterns),
            'predicted_improvements': self._predict_improvements(features, similar_patterns)
        }
        
        logger.info(f"Generated adaptive suggestions with {suggestions['confidence_score']:.2f} confidence")
        return suggestions
    
    def learn_from_optimization(self, 
                              dockerfile_content: str,
                              optimization_result: OptimizationResult,
                              user_feedback: Optional[Dict[str, Any]] = None) -> None:
        """Learn from optimization results and user feedback."""
        # Extract features and create learning context
        features = self.analyze_dockerfile_context(dockerfile_content)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(optimization_result)
        
        # Create learning context
        context = LearningContext(
            dockerfile_type=features.get('base_image_type', 'unknown'),
            project_language=features.get('language_indicators', {}).get('primary', 'unknown'),
            optimization_goals=['security', 'size', 'performance'],  # Could be extracted from user input
            performance_metrics=performance_metrics,
            user_feedback=json.dumps(user_feedback) if user_feedback else None
        )
        
        # Store learning context
        self.learning_history.append(context)
        
        # Update or create optimization pattern
        pattern_id = self._generate_pattern_id(features)
        if pattern_id in self.learned_patterns:
            self._update_existing_pattern(pattern_id, features, performance_metrics)
        else:
            self._create_new_pattern(pattern_id, features, performance_metrics)
        
        # Update user preferences if feedback provided
        if user_feedback:
            self._update_user_preferences(user_feedback)
        
        # Retrain models if enough data
        if len(self.learning_history) % 10 == 0:
            self._retrain_models()
        
        # Save patterns
        self._save_learned_patterns()
        
        logger.info(f"Learned from optimization: pattern {pattern_id}")
    
    def get_adaptive_metrics(self) -> Dict[str, Any]:
        """Get metrics about adaptive learning performance."""
        return {
            'total_patterns': len(self.learned_patterns),
            'learning_history_size': len(self.learning_history),
            'model_accuracy': self._calculate_model_accuracy(),
            'top_performing_patterns': self._get_top_patterns(),
            'adaptation_insights': self._generate_adaptation_insights(),
            'user_preference_summary': self.user_preferences
        }
    
    def _extract_base_image_type(self, dockerfile_content: str) -> str:
        """Extract and categorize base image type."""
        lines = dockerfile_content.strip().split('\n')
        for line in lines:
            if line.strip().upper().startswith('FROM'):
                image = line.split()[1].lower()
                if 'alpine' in image:
                    return 'alpine'
                elif 'ubuntu' in image:
                    return 'ubuntu'
                elif 'debian' in image:
                    return 'debian'
                elif 'node' in image:
                    return 'node'
                elif 'python' in image:
                    return 'python'
                elif 'scratch' in image or 'distroless' in image:
                    return 'minimal'
                else:
                    return 'other'
        return 'unknown'
    
    def _detect_language_patterns(self, dockerfile_content: str) -> Dict[str, Any]:
        """Detect programming language patterns in Dockerfile."""
        language_indicators = {
            'python': ['pip', 'python', 'requirements.txt', 'setup.py', '.py'],
            'node': ['npm', 'node', 'package.json', 'yarn', '.js'],
            'java': ['maven', 'gradle', 'java', '.jar', '.war'],
            'go': ['go build', 'go mod', '.go'],
            'rust': ['cargo', 'rust', '.rs'],
            'dotnet': ['dotnet', '.csproj', '.dll'],
            'php': ['composer', 'php', '.php'],
            'ruby': ['gem', 'ruby', 'Gemfile', '.rb']
        }
        
        detected = {}
        content_lower = dockerfile_content.lower()
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                detected[lang] = score
        
        if detected:
            primary = max(detected, key=detected.get)
            return {'primary': primary, 'scores': detected}
        else:
            return {'primary': 'unknown', 'scores': {}}
    
    def _calculate_complexity_score(self, dockerfile_content: str) -> float:
        """Calculate complexity score based on various factors."""
        lines = dockerfile_content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        factors = {
            'line_count': len(non_empty_lines),
            'run_instructions': dockerfile_content.count('RUN'),
            'copy_instructions': dockerfile_content.count('COPY'),
            'env_instructions': dockerfile_content.count('ENV'),
            'expose_instructions': dockerfile_content.count('EXPOSE'),
            'complexity_keywords': sum(1 for keyword in ['&&', '||', '|', ';'] 
                                     if keyword in dockerfile_content)
        }
        
        # Normalize and weight factors
        score = (
            factors['line_count'] * 0.1 +
            factors['run_instructions'] * 0.3 +
            factors['copy_instructions'] * 0.2 +
            factors['env_instructions'] * 0.1 +
            factors['expose_instructions'] * 0.1 +
            factors['complexity_keywords'] * 0.2
        )
        
        return min(score / 10.0, 1.0)  # Normalize to 0-1
    
    def _identify_security_patterns(self, dockerfile_content: str) -> Dict[str, bool]:
        """Identify security-related patterns."""
        content_upper = dockerfile_content.upper()
        return {
            'has_user_instruction': 'USER ' in content_upper,
            'uses_root': 'USER ROOT' in content_upper,
            'has_healthcheck': 'HEALTHCHECK' in content_upper,
            'pins_versions': ':' in dockerfile_content and any(
                char.isdigit() for char in dockerfile_content.split(':')[-1].split()[0]
            ),
            'updates_packages': 'APT-GET UPDATE' in content_upper or 'YUM UPDATE' in content_upper,
            'cleans_cache': 'RM -RF' in content_upper or 'CLEAN' in content_upper
        }
    
    def _extract_performance_indicators(self, dockerfile_content: str) -> Dict[str, bool]:
        """Extract performance-related indicators."""
        content_upper = dockerfile_content.upper()
        return {
            'has_multistage': 'AS ' in content_upper,
            'combines_run_commands': '&&' in dockerfile_content,
            'uses_cache_optimization': '--NO-CACHE' not in content_upper,
            'minimizes_layers': dockerfile_content.count('RUN') < 5,
            'uses_copy_optimization': 'COPY --FROM=' in content_upper
        }
    
    def _detect_package_managers(self, dockerfile_content: str) -> List[str]:
        """Detect package managers used."""
        managers = []
        content_lower = dockerfile_content.lower()
        
        manager_patterns = {
            'apt': ['apt-get', 'apt '],
            'yum': ['yum '],
            'pip': ['pip ', 'pip3 '],
            'npm': ['npm '],
            'yarn': ['yarn '],
            'composer': ['composer '],
            'gem': ['gem '],
            'cargo': ['cargo '],
            'go': ['go get', 'go mod']
        }
        
        for manager, patterns in manager_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                managers.append(manager)
        
        return managers
    
    def _find_similar_patterns(self, features: Dict[str, Any]) -> List[OptimizationPattern]:
        """Find similar optimization patterns."""
        if not self.learned_patterns:
            return []
        
        similar = []
        for pattern in self.learned_patterns.values():
            similarity = self._calculate_feature_similarity(features, pattern.dockerfile_features)
            if similarity > 0.7:  # Threshold for similarity
                similar.append(pattern)
        
        # Sort by similarity and confidence
        similar.sort(key=lambda p: p.confidence_score, reverse=True)
        return similar[:5]  # Return top 5 similar patterns
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, bool) and isinstance(val2, bool):
                similarity_scores.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2), 1)
                similarity_scores.append(1.0 - abs(val1 - val2) / max_val)
            elif isinstance(val1, str) and isinstance(val2, str):
                similarity_scores.append(1.0 if val1 == val2 else 0.5 if val1 in val2 or val2 in val1 else 0.0)
            elif isinstance(val1, list) and isinstance(val2, list):
                intersection = len(set(val1) & set(val2))
                union = len(set(val1) | set(val2))
                similarity_scores.append(intersection / union if union > 0 else 0.0)
            else:
                similarity_scores.append(0.5)  # Unknown type, neutral score
        
        return sum(similarity_scores) / len(similarity_scores)
    
    def _select_primary_strategy(self, features: Dict[str, Any], similar_patterns: List[OptimizationPattern], user_goals: List[str]) -> str:
        """Select the primary optimization strategy."""
        if similar_patterns:
            # Use the most successful pattern's strategy
            return similar_patterns[0].optimization_strategy
        
        # Fallback to feature-based strategy selection
        if features.get('complexity_score', 0) > 0.7:
            return 'comprehensive_optimization'
        elif 'security' in user_goals and not features.get('security_patterns', {}).get('has_user_instruction'):
            return 'security_focused'
        elif 'size' in user_goals and features.get('layer_count', 0) > 5:
            return 'size_reduction'
        elif 'performance' in user_goals:
            return 'performance_optimization'
        else:
            return 'balanced_optimization'
    
    def _generate_adaptive_optimizations(self, features: Dict[str, Any], similar_patterns: List[OptimizationPattern]) -> List[Dict[str, Any]]:
        """Generate adaptive optimization suggestions."""
        optimizations = []
        
        # Base optimizations from patterns
        if similar_patterns:
            for pattern in similar_patterns[:3]:  # Top 3 patterns
                optimizations.append({
                    'type': 'learned_pattern',
                    'strategy': pattern.optimization_strategy,
                    'confidence': pattern.confidence_score,
                    'description': f"Apply learned optimization pattern (used {pattern.usage_count} times)"
                })
        
        # Feature-based optimizations
        if features.get('base_image_type') == 'ubuntu' and features.get('complexity_score', 0) < 0.5:
            optimizations.append({
                'type': 'base_image_optimization',
                'strategy': 'suggest_alpine',
                'confidence': 0.8,
                'description': 'Consider switching to Alpine Linux for smaller image size'
            })
        
        if not features.get('performance_indicators', {}).get('has_multistage') and features.get('complexity_score', 0) > 0.6:
            optimizations.append({
                'type': 'architecture_optimization',
                'strategy': 'multistage_build',
                'confidence': 0.9,
                'description': 'Implement multi-stage build for better layer caching and size reduction'
            })
        
        if not features.get('security_patterns', {}).get('has_user_instruction'):
            optimizations.append({
                'type': 'security_optimization',
                'strategy': 'add_non_root_user',
                'confidence': 0.95,
                'description': 'Add non-root user for security best practices'
            })
        
        return optimizations
    
    def _calculate_suggestion_confidence(self, similar_patterns: List[OptimizationPattern]) -> float:
        """Calculate confidence score for suggestions."""
        if not similar_patterns:
            return 0.5  # Neutral confidence without patterns
        
        # Weight by pattern confidence and usage
        weighted_confidence = sum(
            pattern.confidence_score * min(pattern.usage_count / 10.0, 1.0)
            for pattern in similar_patterns
        )
        
        return min(weighted_confidence / len(similar_patterns), 1.0)
    
    def _extract_learning_insights(self, similar_patterns: List[OptimizationPattern]) -> List[str]:
        """Extract insights from learning patterns."""
        insights = []
        
        if similar_patterns:
            most_used = max(similar_patterns, key=lambda p: p.usage_count)
            insights.append(f"Most successful strategy: {most_used.optimization_strategy} (used {most_used.usage_count} times)")
            
            avg_confidence = sum(p.confidence_score for p in similar_patterns) / len(similar_patterns)
            insights.append(f"Average pattern confidence: {avg_confidence:.2f}")
            
            recent_patterns = [p for p in similar_patterns if (datetime.now() - p.last_used).days < 30]
            if recent_patterns:
                insights.append(f"Found {len(recent_patterns)} recently used patterns")
        else:
            insights.append("No similar patterns found - this is a new optimization scenario")
        
        return insights
    
    def _predict_improvements(self, features: Dict[str, Any], similar_patterns: List[OptimizationPattern]) -> Dict[str, float]:
        """Predict potential improvements based on patterns."""
        if not similar_patterns:
            return {'size_reduction': 0.2, 'security_improvement': 0.3, 'performance_gain': 0.15}
        
        # Average improvements from similar patterns
        avg_metrics = defaultdict(list)
        for pattern in similar_patterns:
            for metric, value in pattern.success_metrics.items():
                avg_metrics[metric].append(value)
        
        predictions = {}
        for metric, values in avg_metrics.items():
            predictions[metric] = sum(values) / len(values)
        
        return predictions
    
    def _calculate_performance_metrics(self, optimization_result: OptimizationResult) -> Dict[str, float]:
        """Calculate performance metrics from optimization result."""
        metrics = {}
        
        # Size reduction calculation
        if hasattr(optimization_result, 'original_size') and hasattr(optimization_result, 'optimized_size'):
            try:
                original_mb = float(optimization_result.original_size.replace('MB', '').replace('~', ''))
                optimized_mb = float(optimization_result.optimized_size.replace('MB', '').replace('~', ''))
                metrics['size_reduction'] = (original_mb - optimized_mb) / original_mb
            except (ValueError, AttributeError):
                metrics['size_reduction'] = 0.0
        
        # Security improvements
        metrics['security_improvements'] = len(optimization_result.security_fixes) if hasattr(optimization_result, 'security_fixes') else 0.0
        
        # Layer optimizations
        metrics['layer_optimizations'] = len(optimization_result.layer_optimizations) if hasattr(optimization_result, 'layer_optimizations') else 0.0
        
        return metrics
    
    def _generate_pattern_id(self, features: Dict[str, Any]) -> str:
        """Generate a unique pattern ID based on features."""
        key_features = [
            features.get('base_image_type', 'unknown'),
            features.get('language_indicators', {}).get('primary', 'unknown'),
            str(int(features.get('complexity_score', 0) * 10))
        ]
        return '_'.join(key_features)
    
    def _update_existing_pattern(self, pattern_id: str, features: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Update an existing optimization pattern."""
        pattern = self.learned_patterns[pattern_id]
        pattern.usage_count += 1
        pattern.last_used = datetime.now()
        
        # Update success metrics with exponential moving average
        alpha = 0.3  # Learning rate
        for metric, value in metrics.items():
            if metric in pattern.success_metrics:
                pattern.success_metrics[metric] = (1 - alpha) * pattern.success_metrics[metric] + alpha * value
            else:
                pattern.success_metrics[metric] = value
        
        # Update confidence based on usage and success
        pattern.confidence_score = min(
            0.5 + (pattern.usage_count * 0.05) + (sum(pattern.success_metrics.values()) * 0.1),
            1.0
        )
    
    def _create_new_pattern(self, pattern_id: str, features: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Create a new optimization pattern."""
        strategy = self._infer_strategy_from_features(features)
        
        pattern = OptimizationPattern(
            pattern_id=pattern_id,
            dockerfile_features=features,
            optimization_strategy=strategy,
            success_metrics=metrics,
            usage_count=1,
            confidence_score=0.5  # Initial confidence
        )
        
        self.learned_patterns[pattern_id] = pattern
    
    def _infer_strategy_from_features(self, features: Dict[str, Any]) -> str:
        """Infer optimization strategy from features."""
        if features.get('complexity_score', 0) > 0.7:
            return 'comprehensive_optimization'
        elif not features.get('security_patterns', {}).get('has_user_instruction'):
            return 'security_focused'
        elif features.get('layer_count', 0) > 5:
            return 'size_reduction'
        else:
            return 'balanced_optimization'
    
    def _update_user_preferences(self, feedback: Dict[str, Any]) -> None:
        """Update user preferences based on feedback."""
        for key, value in feedback.items():
            if key in self.user_preferences:
                # Exponential moving average for numerical values
                if isinstance(value, (int, float)):
                    self.user_preferences[key] = 0.7 * self.user_preferences[key] + 0.3 * value
                else:
                    self.user_preferences[key] = value
            else:
                self.user_preferences[key] = value
    
    def _retrain_models(self) -> None:
        """Retrain machine learning models with new data."""
        if len(self.learning_history) < 10:
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) > 0:
                # Retrain classifier
                self.pattern_classifier.fit(X, y)
                logger.info("Successfully retrained adaptive models")
        except Exception as e:
            logger.warning(f"Failed to retrain models: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from learning history."""
        # This is a simplified implementation
        # In practice, you'd convert features to numerical vectors
        X, y = [], []
        
        for context in self.learning_history[-100:]:  # Use last 100 examples
            # Convert features to vector (simplified)
            feature_vector = [
                hash(context.dockerfile_type) % 1000,
                hash(context.project_language) % 1000,
                len(context.optimization_goals),
                context.performance_metrics.get('size_reduction', 0),
                context.performance_metrics.get('security_improvements', 0)
            ]
            
            X.append(feature_vector)
            y.append(hash(context.dockerfile_type) % 5)  # Simplified target
        
        return np.array(X), np.array(y)
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate current model accuracy."""
        if len(self.learning_history) < 10:
            return 0.0
        
        try:
            X, y = self._prepare_training_data()
            if len(X) > 0:
                predictions = self.pattern_classifier.predict(X)
                return accuracy_score(y, predictions)
        except Exception:
            pass
        
        return 0.0
    
    def _get_top_patterns(self) -> List[Dict[str, Any]]:
        """Get top performing patterns."""
        patterns = list(self.learned_patterns.values())
        patterns.sort(key=lambda p: p.confidence_score * p.usage_count, reverse=True)
        
        return [
            {
                'pattern_id': p.pattern_id,
                'strategy': p.optimization_strategy,
                'usage_count': p.usage_count,
                'confidence': p.confidence_score
            }
            for p in patterns[:5]
        ]
    
    def _generate_adaptation_insights(self) -> List[str]:
        """Generate insights about adaptation performance."""
        insights = []
        
        if self.learned_patterns:
            total_usage = sum(p.usage_count for p in self.learned_patterns.values())
            avg_confidence = sum(p.confidence_score for p in self.learned_patterns.values()) / len(self.learned_patterns)
            
            insights.append(f"Total pattern applications: {total_usage}")
            insights.append(f"Average pattern confidence: {avg_confidence:.2f}")
            
            recent_patterns = [p for p in self.learned_patterns.values() 
                             if (datetime.now() - p.last_used).days < 7]
            insights.append(f"Active patterns (last 7 days): {len(recent_patterns)}")
        
        return insights
    
    def _load_learned_patterns(self) -> None:
        """Load learned patterns from cache."""
        patterns_file = self.cache_dir / "learned_patterns.pkl"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'rb') as f:
                    data = pickle.load(f)
                    self.learned_patterns = data.get('patterns', {})
                    self.user_preferences = data.get('preferences', {})
                logger.info(f"Loaded {len(self.learned_patterns)} learned patterns")
            except Exception as e:
                logger.warning(f"Failed to load learned patterns: {e}")
    
    def _save_learned_patterns(self) -> None:
        """Save learned patterns to cache."""
        patterns_file = self.cache_dir / "learned_patterns.pkl"
        try:
            with open(patterns_file, 'wb') as f:
                data = {
                    'patterns': self.learned_patterns,
                    'preferences': self.user_preferences,
                    'timestamp': datetime.now()
                }
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save learned patterns: {e}")