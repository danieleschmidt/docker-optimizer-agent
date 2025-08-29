"""Simple API server for Docker Optimizer Agent."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .optimizer import DockerfileOptimizer
from .simple_health_check import SimpleHealthChecker
from .simple_metrics import SimpleMetricsCollector

logger = logging.getLogger(__name__)


class SimpleAPIServer:
    """Simple REST API server for Docker optimization."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the API server. Install with: pip install flask")
        
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.optimizer = DockerfileOptimizer()
        self.health_checker = SimpleHealthChecker()
        self.metrics = SimpleMetricsCollector()
        
        self._setup_routes()
        self._setup_health_checks()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            try:
                health_status = self.health_checker.get_system_health()
                checks = self.health_checker.run_checks()
                
                return jsonify({
                    'status': 'healthy' if self.health_checker.is_healthy() else 'unhealthy',
                    'timestamp': time.time(),
                    'system': {
                        'status': health_status.status,
                        'message': health_status.message,
                        'details': health_status.details
                    },
                    'checks': [
                        {
                            'component': check.component,
                            'status': check.status,
                            'message': check.message
                        } for check in checks
                    ]
                })
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            """Get optimization metrics."""
            try:
                return jsonify(self.metrics.get_summary())
            except Exception as e:
                logger.error(f"Metrics retrieval failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/optimize', methods=['POST'])
        def optimize_dockerfile():
            """Optimize a Dockerfile."""
            start_time = time.time()
            
            try:
                data = request.get_json()
                if not data or 'dockerfile' not in data:
                    return jsonify({'error': 'dockerfile content is required'}), 400
                
                dockerfile_content = data['dockerfile']
                if not dockerfile_content.strip():
                    return jsonify({'error': 'dockerfile content cannot be empty'}), 400
                
                # Optimize the Dockerfile
                result = self.optimizer.optimize_dockerfile(dockerfile_content)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.record_timing('api_optimization', processing_time)
                self.metrics.increment('api_requests')
                self.metrics.increment('dockerfiles_optimized')
                
                # Format response
                response = {
                    'success': True,
                    'original_size': result.original_size,
                    'optimized_size': result.optimized_size,
                    'optimized_dockerfile': result.optimized_dockerfile,
                    'explanation': result.explanation,
                    'security_fixes': len(result.security_fixes),
                    'layer_optimizations': len(result.layer_optimizations),
                    'processing_time': processing_time
                }
                
                return jsonify(response)
                
            except Exception as e:
                self.metrics.increment('api_errors')
                logger.error(f"Optimization failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_dockerfile():
            """Analyze a Dockerfile without optimization."""
            try:
                data = request.get_json()
                if not data or 'dockerfile' not in data:
                    return jsonify({'error': 'dockerfile content is required'}), 400
                
                dockerfile_content = data['dockerfile']
                analysis = self.optimizer.analyze_dockerfile(dockerfile_content)
                
                self.metrics.increment('api_requests')
                self.metrics.increment('analysis_requests')
                
                response = {
                    'success': True,
                    'base_image': analysis.base_image,
                    'total_layers': analysis.total_layers,
                    'security_issues': analysis.security_issues,
                    'optimization_opportunities': analysis.optimization_opportunities,
                    'estimated_size': analysis.estimated_size
                }
                
                return jsonify(response)
                
            except Exception as e:
                self.metrics.increment('api_errors')
                logger.error(f"Analysis failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/', methods=['GET'])
        def index():
            """API information endpoint."""
            return jsonify({
                'name': 'Docker Optimizer Agent API',
                'version': '1.0.0',
                'endpoints': {
                    'GET /': 'API information',
                    'GET /health': 'Health check',
                    'GET /metrics': 'Get metrics',
                    'POST /optimize': 'Optimize Dockerfile',
                    'POST /analyze': 'Analyze Dockerfile'
                }
            })
    
    def _setup_health_checks(self):
        """Setup health check functions."""
        def optimizer_health():
            try:
                # Simple test optimization
                test_dockerfile = "FROM alpine:3.18\nRUN echo 'test'"
                result = self.optimizer.analyze_dockerfile(test_dockerfile)
                return result.base_image == "alpine:3.18"
            except:
                return False
        
        self.health_checker.add_check('optimizer', optimizer_health, 60)
    
    def run(self, debug: bool = False):
        """Start the API server."""
        logger.info(f"Starting Docker Optimizer API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
    
    def get_app(self):
        """Get the Flask app instance for testing."""
        return self.app