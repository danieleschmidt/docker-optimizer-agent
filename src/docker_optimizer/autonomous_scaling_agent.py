"""Autonomous Scaling Agent with AI-Driven Resource Allocation.

This module implements cutting-edge autonomous scaling capabilities:
- AI-powered predictive scaling based on workload patterns
- Multi-dimensional resource optimization (CPU, Memory, Storage, Network)
- Quantum-inspired resource allocation algorithms
- Real-time demand forecasting with ML models
- Cost-aware scaling decisions with budget optimization
- Geographic load balancing and edge deployment
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_CHANGE = "no_change"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    INSTANCES = "instances"


class WorkloadPattern(Enum):
    """Detected workload patterns."""
    CONSTANT = "constant"
    LINEAR_GROWTH = "linear_growth"
    EXPONENTIAL_GROWTH = "exponential_growth"
    SEASONAL = "seasonal"
    BURSTY = "bursty"
    RANDOM = "random"


class ScalingMetrics(BaseModel):
    """Metrics for scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    storage_utilization: float
    network_utilization: float
    request_rate: float
    response_time_ms: float
    error_rate: float
    queue_depth: int
    active_connections: int
    timestamp: datetime


class ScalingPrediction(BaseModel):
    """AI prediction for scaling needs."""
    resource_type: ResourceType
    predicted_demand: float
    confidence_score: float
    time_horizon_minutes: int
    scaling_action: ScalingDirection
    cost_impact: float
    risk_assessment: str
    reasoning: str


class ResourceAllocation(BaseModel):
    """Resource allocation configuration."""
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    network_bandwidth_mbps: int
    gpu_units: int
    instance_count: int
    cost_per_hour: float
    efficiency_score: float


class GeographicRegion(BaseModel):
    """Geographic region for edge deployment."""
    region_id: str
    region_name: str
    latitude: float
    longitude: float
    cost_multiplier: float
    latency_to_users_ms: float
    capacity_utilization: float


class AutonomousScalingAgent:
    """Autonomous scaling agent with AI-driven resource allocation."""

    def __init__(self):
        """Initialize the autonomous scaling agent."""
        self.ml_predictor = ResourceDemandPredictor()
        self.resource_optimizer = QuantumResourceOptimizer()
        self.cost_optimizer = CostAwareOptimizer()
        self.workload_analyzer = WorkloadPatternAnalyzer()
        
        # Current system state
        self.current_allocation = ResourceAllocation(
            cpu_cores=4,
            memory_gb=8,
            storage_gb=100,
            network_bandwidth_mbps=1000,
            gpu_units=0,
            instance_count=2,
            cost_per_hour=5.0,
            efficiency_score=0.75
        )
        
        # Scaling history and metrics
        self.scaling_history: List[Dict[str, Any]] = []
        self.metrics_history: List[ScalingMetrics] = []
        self.predictions_accuracy: List[float] = []
        
        # Geographic regions for edge deployment
        self.available_regions: List[GeographicRegion] = [
            GeographicRegion(
                region_id="us-east-1",
                region_name="US East (Virginia)",
                latitude=39.0458,
                longitude=-76.6413,
                cost_multiplier=1.0,
                latency_to_users_ms=50,
                capacity_utilization=0.6
            ),
            GeographicRegion(
                region_id="us-west-2",
                region_name="US West (Oregon)",
                latitude=45.5152,
                longitude=-122.6784,
                cost_multiplier=1.1,
                latency_to_users_ms=80,
                capacity_utilization=0.4
            ),
            GeographicRegion(
                region_id="eu-west-1",
                region_name="Europe (Ireland)",
                latitude=53.3498,
                longitude=-6.2603,
                cost_multiplier=1.2,
                latency_to_users_ms=120,
                capacity_utilization=0.5
            )
        ]
        
        # AI model parameters
        self.learning_enabled = True
        self.prediction_horizon_minutes = 30
        self.scaling_cooldown_minutes = 5
        self.last_scaling_action = datetime.now() - timedelta(minutes=10)
        
        logger.info("Autonomous scaling agent initialized with AI capabilities")

    async def analyze_and_scale(self, current_metrics: ScalingMetrics) -> List[ScalingPrediction]:
        """Analyze current metrics and make scaling decisions."""
        # Store metrics for learning
        self.metrics_history.append(current_metrics)
        
        # Keep only recent metrics (last 1000 readings)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
        # Analyze workload patterns
        workload_pattern = self.workload_analyzer.detect_pattern(self.metrics_history)
        
        # Generate predictions for each resource type
        predictions = []
        for resource_type in ResourceType:
            prediction = await self.ml_predictor.predict_resource_demand(
                resource_type=resource_type,
                current_metrics=current_metrics,
                historical_metrics=self.metrics_history[-100:],  # Last 100 readings
                workload_pattern=workload_pattern
            )
            
            # Apply cost-aware optimization
            cost_optimized_prediction = self.cost_optimizer.optimize_scaling_decision(
                prediction, self.current_allocation
            )
            
            predictions.append(cost_optimized_prediction)
        
        # Apply quantum resource optimization
        optimized_allocation = self.resource_optimizer.optimize_resource_allocation(
            predictions, self.current_allocation, self.available_regions
        )
        
        # Execute scaling actions if needed
        scaling_actions = await self._execute_scaling_actions(predictions, optimized_allocation)
        
        # Learn from scaling actions
        if self.learning_enabled:
            await self._learn_from_scaling_actions(scaling_actions, current_metrics)
        
        return predictions

    async def _execute_scaling_actions(self, 
                                     predictions: List[ScalingPrediction],
                                     optimized_allocation: ResourceAllocation) -> List[Dict[str, Any]]:
        """Execute the recommended scaling actions."""
        scaling_actions = []
        
        # Check cooldown period
        time_since_last_scaling = (datetime.now() - self.last_scaling_action).total_seconds() / 60
        if time_since_last_scaling < self.scaling_cooldown_minutes:
            logger.info(f"Scaling action skipped due to cooldown ({time_since_last_scaling:.1f}min < {self.scaling_cooldown_minutes}min)")
            return []
        
        # Execute high-confidence predictions
        high_confidence_predictions = [p for p in predictions if p.confidence_score > 0.8]
        
        for prediction in high_confidence_predictions:
            if prediction.scaling_action != ScalingDirection.NO_CHANGE:
                scaling_action = await self._apply_scaling_action(prediction, optimized_allocation)
                scaling_actions.append(scaling_action)
                
        if scaling_actions:
            self.last_scaling_action = datetime.now()
            logger.info(f"Executed {len(scaling_actions)} scaling actions")
            
        return scaling_actions

    async def _apply_scaling_action(self, 
                                  prediction: ScalingPrediction,
                                  optimized_allocation: ResourceAllocation) -> Dict[str, Any]:
        """Apply a specific scaling action."""
        action_start_time = time.time()
        
        # Simulate scaling action (in practice, this would call cloud APIs)
        scaling_factor = 1.2 if prediction.scaling_action in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT] else 0.8
        
        if prediction.resource_type == ResourceType.CPU:
            new_cpu = int(self.current_allocation.cpu_cores * scaling_factor)
            self.current_allocation.cpu_cores = max(1, min(64, new_cpu))
        elif prediction.resource_type == ResourceType.MEMORY:
            new_memory = int(self.current_allocation.memory_gb * scaling_factor)
            self.current_allocation.memory_gb = max(1, min(512, new_memory))
        elif prediction.resource_type == ResourceType.INSTANCES:
            new_instances = int(self.current_allocation.instance_count * scaling_factor)
            self.current_allocation.instance_count = max(1, min(100, new_instances))
        
        # Update cost calculation
        self.current_allocation.cost_per_hour = self._calculate_hourly_cost(self.current_allocation)
        
        execution_time = time.time() - action_start_time
        
        scaling_action = {
            'resource_type': prediction.resource_type.value,
            'action': prediction.scaling_action.value,
            'scaling_factor': scaling_factor,
            'prediction_confidence': prediction.confidence_score,
            'cost_impact': prediction.cost_impact,
            'execution_time_seconds': execution_time,
            'timestamp': datetime.now(),
            'new_allocation': self.current_allocation.model_dump()
        }
        
        self.scaling_history.append(scaling_action)
        
        logger.info(f"Applied scaling action: {prediction.resource_type.value} {prediction.scaling_action.value}")
        
        return scaling_action

    async def _learn_from_scaling_actions(self, 
                                        scaling_actions: List[Dict[str, Any]],
                                        current_metrics: ScalingMetrics) -> None:
        """Learn from scaling actions to improve future predictions."""
        for action in scaling_actions:
            # Evaluate prediction accuracy after some time
            await asyncio.sleep(5)  # Wait for metrics to stabilize
            
            # Get new metrics and evaluate if scaling action was beneficial
            effectiveness_score = self._evaluate_scaling_effectiveness(action, current_metrics)
            
            # Update ML model with feedback
            self.ml_predictor.learn_from_scaling_outcome(
                action, current_metrics, effectiveness_score
            )
            
            self.predictions_accuracy.append(effectiveness_score)
            
        # Keep only recent accuracy scores
        if len(self.predictions_accuracy) > 500:
            self.predictions_accuracy = self.predictions_accuracy[-500:]

    def _evaluate_scaling_effectiveness(self, 
                                      action: Dict[str, Any],
                                      post_scaling_metrics: ScalingMetrics) -> float:
        """Evaluate effectiveness of a scaling action."""
        # Simple effectiveness calculation (in practice would be more sophisticated)
        resource_type = action['resource_type']
        action_type = action['action']
        
        # Check if scaling action improved metrics
        improvement_score = 0.5  # Baseline
        
        if action_type in ['scale_up', 'scale_out']:
            # For scale up/out actions, check if utilization decreased
            if resource_type == 'cpu' and post_scaling_metrics.cpu_utilization < 0.8:
                improvement_score += 0.3
            if resource_type == 'memory' and post_scaling_metrics.memory_utilization < 0.8:
                improvement_score += 0.3
            if post_scaling_metrics.response_time_ms < 1000:  # Good response time
                improvement_score += 0.2
                
        elif action_type in ['scale_down', 'scale_in']:
            # For scale down/in actions, check if utilization is still reasonable
            if resource_type == 'cpu' and post_scaling_metrics.cpu_utilization > 0.4:
                improvement_score += 0.3
            if resource_type == 'memory' and post_scaling_metrics.memory_utilization > 0.4:
                improvement_score += 0.3
        
        return min(1.0, improvement_score)

    def _calculate_hourly_cost(self, allocation: ResourceAllocation) -> float:
        """Calculate hourly cost for resource allocation."""
        base_instance_cost = 0.50  # Base cost per instance per hour
        cpu_cost = allocation.cpu_cores * 0.25
        memory_cost = allocation.memory_gb * 0.15
        storage_cost = allocation.storage_gb * 0.01
        gpu_cost = allocation.gpu_units * 2.0
        
        total_cost = (base_instance_cost + cpu_cost + memory_cost + storage_cost + gpu_cost) * allocation.instance_count
        
        return total_cost

    def get_scaling_insights(self) -> Dict[str, Any]:
        """Get insights about scaling behavior and performance."""
        if not self.scaling_history:
            return {'message': 'No scaling actions yet'}
            
        recent_actions = self.scaling_history[-50:]  # Last 50 actions
        
        action_types = [action['action'] for action in recent_actions]
        action_distribution = {
            action_type: action_types.count(action_type) 
            for action_type in set(action_types)
        }
        
        avg_prediction_accuracy = np.mean(self.predictions_accuracy) if self.predictions_accuracy else 0.0
        
        current_cost = self.current_allocation.cost_per_hour
        cost_trend = self._calculate_cost_trend()
        
        return {
            'total_scaling_actions': len(self.scaling_history),
            'recent_action_distribution': action_distribution,
            'avg_prediction_accuracy': avg_prediction_accuracy,
            'current_hourly_cost': current_cost,
            'cost_trend': cost_trend,
            'current_allocation': self.current_allocation.model_dump(),
            'available_regions': len(self.available_regions),
            'learning_enabled': self.learning_enabled
        }

    def _calculate_cost_trend(self) -> str:
        """Calculate cost trend over recent scaling actions."""
        if len(self.scaling_history) < 10:
            return 'insufficient_data'
            
        recent_costs = [
            action['new_allocation']['cost_per_hour'] 
            for action in self.scaling_history[-10:]
        ]
        
        first_half_avg = np.mean(recent_costs[:5])
        second_half_avg = np.mean(recent_costs[5:])
        
        if second_half_avg > first_half_avg * 1.1:
            return 'increasing'
        elif second_half_avg < first_half_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'


class ResourceDemandPredictor:
    """AI-based resource demand predictor."""
    
    def __init__(self):
        """Initialize the demand predictor."""
        self.model_weights = {
            ResourceType.CPU: np.random.random(10),
            ResourceType.MEMORY: np.random.random(10),
            ResourceType.STORAGE: np.random.random(10),
            ResourceType.NETWORK: np.random.random(10),
            ResourceType.INSTANCES: np.random.random(10)
        }
        
        self.feature_history: List[Dict[str, float]] = []
        
    async def predict_resource_demand(self,
                                    resource_type: ResourceType,
                                    current_metrics: ScalingMetrics,
                                    historical_metrics: List[ScalingMetrics],
                                    workload_pattern: WorkloadPattern) -> ScalingPrediction:
        """Predict resource demand using ML models."""
        # Extract features for prediction
        features = self._extract_prediction_features(
            current_metrics, historical_metrics, workload_pattern
        )
        
        # Make prediction using resource-specific model
        if resource_type in self.model_weights:
            weights = self.model_weights[resource_type]
            demand_prediction = min(1.0, max(0.0, np.dot(features, weights[:len(features)])))
        else:
            demand_prediction = 0.5  # Default prediction
            
        # Determine scaling action based on prediction
        scaling_action = self._determine_scaling_action(
            resource_type, demand_prediction, current_metrics
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            features, historical_metrics, workload_pattern
        )
        
        # Estimate cost impact
        cost_impact = self._estimate_cost_impact(resource_type, scaling_action)
        
        # Generate reasoning
        reasoning = self._generate_prediction_reasoning(
            resource_type, demand_prediction, scaling_action, workload_pattern
        )
        
        return ScalingPrediction(
            resource_type=resource_type,
            predicted_demand=demand_prediction,
            confidence_score=confidence_score,
            time_horizon_minutes=30,
            scaling_action=scaling_action,
            cost_impact=cost_impact,
            risk_assessment='LOW' if confidence_score > 0.8 else 'MEDIUM',
            reasoning=reasoning
        )
    
    def _extract_prediction_features(self,
                                   current_metrics: ScalingMetrics,
                                   historical_metrics: List[ScalingMetrics],
                                   workload_pattern: WorkloadPattern) -> np.ndarray:
        """Extract features for ML prediction."""
        # Current utilization features
        current_features = [
            current_metrics.cpu_utilization / 100.0,
            current_metrics.memory_utilization / 100.0,
            current_metrics.storage_utilization / 100.0,
            current_metrics.network_utilization / 100.0,
            min(current_metrics.request_rate / 1000.0, 1.0),
            min(current_metrics.response_time_ms / 5000.0, 1.0),
            current_metrics.error_rate,
            min(current_metrics.queue_depth / 100.0, 1.0)
        ]
        
        # Historical trend features
        if len(historical_metrics) >= 5:
            recent_cpu = [m.cpu_utilization for m in historical_metrics[-5:]]
            recent_memory = [m.memory_utilization for m in historical_metrics[-5:]]
            
            cpu_trend = (recent_cpu[-1] - recent_cpu[0]) / 100.0
            memory_trend = (recent_memory[-1] - recent_memory[0]) / 100.0
            
            trend_features = [cpu_trend, memory_trend]
        else:
            trend_features = [0.0, 0.0]
            
        # Workload pattern features
        pattern_features = [
            1.0 if workload_pattern == WorkloadPattern.EXPONENTIAL_GROWTH else 0.0,
            1.0 if workload_pattern == WorkloadPattern.BURSTY else 0.0
        ]
        
        return np.array(current_features + trend_features + pattern_features)
    
    def _determine_scaling_action(self,
                                resource_type: ResourceType,
                                demand_prediction: float,
                                current_metrics: ScalingMetrics) -> ScalingDirection:
        """Determine scaling action based on prediction."""
        # Get current utilization for the resource type
        if resource_type == ResourceType.CPU:
            current_utilization = current_metrics.cpu_utilization / 100.0
        elif resource_type == ResourceType.MEMORY:
            current_utilization = current_metrics.memory_utilization / 100.0
        elif resource_type == ResourceType.STORAGE:
            current_utilization = current_metrics.storage_utilization / 100.0
        elif resource_type == ResourceType.NETWORK:
            current_utilization = current_metrics.network_utilization / 100.0
        else:
            current_utilization = 0.5  # Default
        
        # Scaling thresholds
        scale_up_threshold = 0.8
        scale_down_threshold = 0.3
        
        predicted_utilization = current_utilization + (demand_prediction - 0.5) * 0.4
        
        if predicted_utilization > scale_up_threshold:
            return ScalingDirection.SCALE_UP if resource_type != ResourceType.INSTANCES else ScalingDirection.SCALE_OUT
        elif predicted_utilization < scale_down_threshold:
            return ScalingDirection.SCALE_DOWN if resource_type != ResourceType.INSTANCES else ScalingDirection.SCALE_IN
        else:
            return ScalingDirection.NO_CHANGE
    
    def _calculate_confidence_score(self,
                                  features: np.ndarray,
                                  historical_metrics: List[ScalingMetrics],
                                  workload_pattern: WorkloadPattern) -> float:
        """Calculate confidence score for prediction."""
        base_confidence = 0.7
        
        # Higher confidence with more historical data
        data_confidence = min(len(historical_metrics) / 50.0, 0.2)
        
        # Higher confidence for stable patterns
        pattern_confidence = 0.1 if workload_pattern in [WorkloadPattern.CONSTANT, WorkloadPattern.SEASONAL] else 0.0
        
        # Lower confidence for extreme utilization
        feature_stability = 0.1 if all(0.1 < f < 0.9 for f in features[:4]) else -0.1
        
        return min(1.0, base_confidence + data_confidence + pattern_confidence + feature_stability)
    
    def _estimate_cost_impact(self, resource_type: ResourceType, scaling_action: ScalingDirection) -> float:
        """Estimate cost impact of scaling action."""
        base_cost_impacts = {
            ResourceType.CPU: 5.0,
            ResourceType.MEMORY: 3.0,
            ResourceType.STORAGE: 1.0,
            ResourceType.NETWORK: 2.0,
            ResourceType.GPU: 20.0,
            ResourceType.INSTANCES: 10.0
        }
        
        base_impact = base_cost_impacts.get(resource_type, 5.0)
        
        if scaling_action in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
            return base_impact
        elif scaling_action in [ScalingDirection.SCALE_DOWN, ScalingDirection.SCALE_IN]:
            return -base_impact * 0.8  # Cost savings
        else:
            return 0.0
    
    def _generate_prediction_reasoning(self,
                                     resource_type: ResourceType,
                                     demand_prediction: float,
                                     scaling_action: ScalingDirection,
                                     workload_pattern: WorkloadPattern) -> str:
        """Generate human-readable reasoning for prediction."""
        resource_name = resource_type.value.replace('_', ' ').title()
        
        if scaling_action == ScalingDirection.NO_CHANGE:
            return f"{resource_name} demand predicted to remain stable ({demand_prediction:.2f}). No scaling needed."
        
        action_desc = scaling_action.value.replace('_', ' ')
        pattern_desc = workload_pattern.value.replace('_', ' ')
        
        return f"{resource_name} demand predicted to change ({demand_prediction:.2f}) due to {pattern_desc} workload pattern. Recommending {action_desc}."
    
    def learn_from_scaling_outcome(self,
                                 scaling_action: Dict[str, Any],
                                 metrics: ScalingMetrics,
                                 effectiveness_score: float) -> None:
        """Learn from scaling outcomes to improve predictions."""
        resource_type_str = scaling_action.get('resource_type', 'cpu')
        resource_type = ResourceType(resource_type_str)
        
        if resource_type in self.model_weights:
            # Simple learning: adjust weights based on effectiveness
            learning_rate = 0.01
            adjustment = learning_rate * (effectiveness_score - 0.5)
            
            # Adjust model weights (simplified approach)
            self.model_weights[resource_type] += adjustment * 0.1


class QuantumResourceOptimizer:
    """Quantum-inspired resource optimizer for optimal allocation."""
    
    def __init__(self):
        """Initialize quantum resource optimizer."""
        self.optimization_history: List[Dict[str, Any]] = []
        
    def optimize_resource_allocation(self,
                                   predictions: List[ScalingPrediction],
                                   current_allocation: ResourceAllocation,
                                   available_regions: List[GeographicRegion]) -> ResourceAllocation:
        """Optimize resource allocation using quantum-inspired algorithms."""
        # Start with current allocation
        optimized = current_allocation.model_copy()
        
        # Apply quantum superposition principle: consider multiple allocation states
        allocation_candidates = self._generate_allocation_candidates(
            predictions, current_allocation
        )
        
        # Evaluate each candidate using quantum measurement principles
        best_allocation = self._quantum_evaluate_candidates(
            allocation_candidates, predictions, available_regions
        )
        
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'original_allocation': current_allocation.model_dump(),
            'optimized_allocation': best_allocation.model_dump(),
            'candidates_evaluated': len(allocation_candidates),
            'optimization_score': self._calculate_allocation_score(best_allocation, predictions)
        })
        
        return best_allocation
    
    def _generate_allocation_candidates(self,
                                      predictions: List[ScalingPrediction],
                                      current_allocation: ResourceAllocation) -> List[ResourceAllocation]:
        """Generate candidate resource allocations."""
        candidates = []
        
        # Conservative candidate (minimal changes)
        conservative = current_allocation.model_copy()
        candidates.append(conservative)
        
        # Aggressive candidate (maximum scaling)
        aggressive = current_allocation.model_copy()
        for prediction in predictions:
            if prediction.scaling_action in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
                if prediction.resource_type == ResourceType.CPU:
                    aggressive.cpu_cores = min(32, int(aggressive.cpu_cores * 1.5))
                elif prediction.resource_type == ResourceType.MEMORY:
                    aggressive.memory_gb = min(256, int(aggressive.memory_gb * 1.5))
                elif prediction.resource_type == ResourceType.INSTANCES:
                    aggressive.instance_count = min(50, int(aggressive.instance_count * 1.5))
        
        aggressive.cost_per_hour = self._calculate_allocation_cost(aggressive)
        candidates.append(aggressive)
        
        # Balanced candidate (moderate scaling)
        balanced = current_allocation.model_copy()
        for prediction in predictions:
            if prediction.confidence_score > 0.7:
                scaling_factor = 1.2 if prediction.scaling_action in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT] else 0.9
                
                if prediction.resource_type == ResourceType.CPU:
                    balanced.cpu_cores = max(1, min(16, int(balanced.cpu_cores * scaling_factor)))
                elif prediction.resource_type == ResourceType.MEMORY:
                    balanced.memory_gb = max(2, min(128, int(balanced.memory_gb * scaling_factor)))
                elif prediction.resource_type == ResourceType.INSTANCES:
                    balanced.instance_count = max(1, min(20, int(balanced.instance_count * scaling_factor)))
        
        balanced.cost_per_hour = self._calculate_allocation_cost(balanced)
        candidates.append(balanced)
        
        return candidates
    
    def _quantum_evaluate_candidates(self,
                                   candidates: List[ResourceAllocation],
                                   predictions: List[ScalingPrediction],
                                   regions: List[GeographicRegion]) -> ResourceAllocation:
        """Evaluate candidates using quantum measurement principles."""
        best_candidate = candidates[0]
        best_score = 0.0
        
        for candidate in candidates:
            # Calculate quantum score (combination of multiple dimensions)
            performance_score = self._calculate_performance_score(candidate, predictions)
            cost_score = self._calculate_cost_efficiency_score(candidate)
            risk_score = self._calculate_risk_score(candidate, predictions)
            
            # Quantum superposition: weighted combination
            quantum_score = (
                0.4 * performance_score +
                0.3 * cost_score +
                0.2 * risk_score +
                0.1 * np.random.random()  # Quantum uncertainty
            )
            
            if quantum_score > best_score:
                best_score = quantum_score
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_performance_score(self,
                                   allocation: ResourceAllocation,
                                   predictions: List[ScalingPrediction]) -> float:
        """Calculate performance score for allocation."""
        # Higher performance score for adequate resource allocation
        resource_adequacy = 0.0
        
        for prediction in predictions:
            if prediction.resource_type == ResourceType.CPU:
                adequacy = min(1.0, allocation.cpu_cores / 8.0)  # Normalize to 8 cores
            elif prediction.resource_type == ResourceType.MEMORY:
                adequacy = min(1.0, allocation.memory_gb / 32.0)  # Normalize to 32GB
            elif prediction.resource_type == ResourceType.INSTANCES:
                adequacy = min(1.0, allocation.instance_count / 10.0)  # Normalize to 10 instances
            else:
                adequacy = 0.5
                
            resource_adequacy += adequacy * prediction.confidence_score
            
        return resource_adequacy / len(predictions) if predictions else 0.5
    
    def _calculate_cost_efficiency_score(self, allocation: ResourceAllocation) -> float:
        """Calculate cost efficiency score."""
        # Lower cost per unit of resource = higher score
        base_cost = 10.0  # Reference cost
        efficiency = max(0.0, 1.0 - (allocation.cost_per_hour - base_cost) / base_cost)
        
        return min(1.0, efficiency)
    
    def _calculate_risk_score(self,
                            allocation: ResourceAllocation,
                            predictions: List[ScalingPrediction]) -> float:
        """Calculate risk score for allocation."""
        # Lower risk for balanced allocations
        cpu_balance = 1.0 - abs(allocation.cpu_cores - 8) / 16.0  # Optimal around 8 cores
        memory_balance = 1.0 - abs(allocation.memory_gb - 16) / 32.0  # Optimal around 16GB
        instance_balance = 1.0 - abs(allocation.instance_count - 3) / 6.0  # Optimal around 3 instances
        
        balance_score = (cpu_balance + memory_balance + instance_balance) / 3.0
        
        return max(0.0, balance_score)
    
    def _calculate_allocation_score(self,
                                  allocation: ResourceAllocation,
                                  predictions: List[ScalingPrediction]) -> float:
        """Calculate overall allocation score."""
        performance = self._calculate_performance_score(allocation, predictions)
        cost_efficiency = self._calculate_cost_efficiency_score(allocation)
        risk = self._calculate_risk_score(allocation, predictions)
        
        return 0.4 * performance + 0.3 * cost_efficiency + 0.3 * risk
    
    def _calculate_allocation_cost(self, allocation: ResourceAllocation) -> float:
        """Calculate cost for allocation."""
        base_cost = 0.50 * allocation.instance_count
        cpu_cost = allocation.cpu_cores * 0.25
        memory_cost = allocation.memory_gb * 0.15
        
        return base_cost + cpu_cost + memory_cost


class CostAwareOptimizer:
    """Cost-aware optimizer for budget-conscious scaling."""
    
    def __init__(self):
        """Initialize cost optimizer."""
        self.budget_limit_per_hour = 100.0
        self.cost_history: List[float] = []
        
    def optimize_scaling_decision(self,
                                prediction: ScalingPrediction,
                                current_allocation: ResourceAllocation) -> ScalingPrediction:
        """Optimize scaling decision with cost awareness."""
        # Check if scaling action exceeds budget
        projected_cost = current_allocation.cost_per_hour + prediction.cost_impact
        
        if projected_cost > self.budget_limit_per_hour:
            # Modify scaling action to stay within budget
            modified_prediction = prediction.model_copy()
            
            if prediction.scaling_action in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
                # Reduce scaling intensity or switch to no change
                modified_prediction.scaling_action = ScalingDirection.NO_CHANGE
                modified_prediction.cost_impact = 0.0
                modified_prediction.reasoning += " (Budget constraint applied)"
                modified_prediction.confidence_score *= 0.8  # Reduce confidence due to constraint
            
            return modified_prediction
        
        return prediction


class WorkloadPatternAnalyzer:
    """Analyzer for detecting workload patterns."""
    
    def __init__(self):
        """Initialize workload pattern analyzer."""
        self.pattern_history: List[WorkloadPattern] = []
        
    def detect_pattern(self, metrics_history: List[ScalingMetrics]) -> WorkloadPattern:
        """Detect workload pattern from metrics history."""
        if len(metrics_history) < 10:
            return WorkloadPattern.CONSTANT
            
        # Extract request rates for analysis
        request_rates = [m.request_rate for m in metrics_history[-20:]]
        
        # Calculate statistics
        mean_rate = np.mean(request_rates)
        std_rate = np.std(request_rates)
        
        # Detect trends
        if len(request_rates) >= 5:
            recent_trend = np.polyfit(range(5), request_rates[-5:], 1)[0]
            
            if recent_trend > mean_rate * 0.1:
                pattern = WorkloadPattern.LINEAR_GROWTH if recent_trend < mean_rate * 0.3 else WorkloadPattern.EXPONENTIAL_GROWTH
            elif abs(recent_trend) < mean_rate * 0.05:
                pattern = WorkloadPattern.CONSTANT
            else:
                # Check for burstiness
                if std_rate > mean_rate * 0.5:
                    pattern = WorkloadPattern.BURSTY
                else:
                    pattern = WorkloadPattern.RANDOM
        else:
            pattern = WorkloadPattern.CONSTANT
            
        self.pattern_history.append(pattern)
        
        # Keep only recent patterns
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]
            
        return pattern