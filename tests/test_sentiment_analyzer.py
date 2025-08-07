"""Test suite for DockerfileSentimentAnalyzer."""

import pytest
from docker_optimizer.sentiment_analyzer import (
    DockerfileSentimentAnalyzer,
    SentimentScore,
    FeedbackTone,
    SentimentAnalysis,
    OptimizedFeedback,
)


class TestDockerfileSentimentAnalyzer:
    """Test cases for DockerfileSentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a DockerfileSentimentAnalyzer instance."""
        return DockerfileSentimentAnalyzer()

    def test_sentiment_analyzer_initialization(self, analyzer):
        """Test that sentiment analyzer initializes correctly."""
        assert analyzer is not None
        assert hasattr(analyzer, 'positive_keywords')
        assert hasattr(analyzer, 'negative_keywords')
        assert hasattr(analyzer, 'warning_keywords')
        assert hasattr(analyzer, 'response_templates')

    def test_analyze_positive_sentiment(self, analyzer):
        """Test analysis of positive sentiment feedback."""
        positive_text = "Excellent work! Your Dockerfile is well-optimized and secure."
        
        analysis = analyzer.analyze_sentiment(positive_text)
        
        assert isinstance(analysis, SentimentAnalysis)
        assert analysis.sentiment_score in [SentimentScore.POSITIVE, SentimentScore.VERY_POSITIVE]
        assert analysis.tone == FeedbackTone.ENCOURAGING
        assert analysis.confidence > 0.5
        assert 'excellent' in analysis.emotional_keywords
        assert 'optimized' in analysis.emotional_keywords

    def test_analyze_negative_sentiment(self, analyzer):
        """Test analysis of negative sentiment feedback."""
        negative_text = "This Dockerfile has serious security vulnerabilities and broken dependencies."
        
        analysis = analyzer.analyze_sentiment(negative_text)
        
        assert isinstance(analysis, SentimentAnalysis)
        assert analysis.sentiment_score in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE]
        assert analysis.tone in [FeedbackTone.WARNING, FeedbackTone.CRITICAL]
        assert analysis.confidence > 0.5
        assert 'vulnerable' in analysis.emotional_keywords or 'broken' in analysis.emotional_keywords

    def test_analyze_neutral_sentiment(self, analyzer):
        """Test analysis of neutral sentiment feedback."""
        neutral_text = "The Dockerfile uses standard commands for package installation."
        
        analysis = analyzer.analyze_sentiment(neutral_text)
        
        assert isinstance(analysis, SentimentAnalysis)
        assert analysis.sentiment_score == SentimentScore.NEUTRAL
        assert analysis.tone == FeedbackTone.INFORMATIVE
        assert analysis.confidence >= 0.5

    def test_optimize_feedback_positive(self, analyzer):
        """Test feedback optimization for positive sentiment."""
        positive_feedback = "Great job using specific versions for security!"
        
        optimized = analyzer.optimize_feedback(positive_feedback, "security")
        
        assert isinstance(optimized, OptimizedFeedback)
        assert optimized.original_message == positive_feedback
        assert "🎉" in optimized.optimized_message or "✨" in optimized.optimized_message or "🚀" in optimized.optimized_message
        assert optimized.improvement_category == "security"
        assert optimized.sentiment_analysis.sentiment_score in [
            SentimentScore.POSITIVE, SentimentScore.VERY_POSITIVE
        ]

    def test_optimize_feedback_negative(self, analyzer):
        """Test feedback optimization for negative sentiment."""
        negative_feedback = "Error: Your Dockerfile has broken security configurations."
        
        optimized = analyzer.optimize_feedback(negative_feedback, "security")
        
        assert isinstance(optimized, OptimizedFeedback)
        assert optimized.original_message == negative_feedback
        # Should transform harsh language
        assert "issue" in optimized.optimized_message
        assert "Error" not in optimized.optimized_message or "🚨" in optimized.optimized_message
        assert optimized.improvement_category == "security"

    def test_enhance_message_harsh_language(self, analyzer):
        """Test that harsh technical terms are replaced with friendlier alternatives."""
        harsh_message = "Your Dockerfile is broken and has failed security checks."
        
        analysis = analyzer.analyze_sentiment(harsh_message)
        enhanced = analyzer._enhance_message(harsh_message, analysis)
        
        assert "needs updating" in enhanced
        assert "needs attention" in enhanced
        assert "broken" not in enhanced
        assert "failed" not in enhanced

    def test_generate_suggestions_negative_sentiment(self, analyzer):
        """Test suggestion generation for negative sentiment."""
        suggestions = analyzer._generate_suggestions(
            SentimentScore.VERY_NEGATIVE, 
            ['security', 'broken', 'error']
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any('security' in s.lower() for s in suggestions)
        assert any('actionable' in s.lower() or 'step' in s.lower() for s in suggestions)

    def test_generate_suggestions_neutral_sentiment(self, analyzer):
        """Test suggestion generation for neutral sentiment."""
        suggestions = analyzer._generate_suggestions(
            SentimentScore.NEUTRAL, 
            ['performance']
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any('encouraging' in s.lower() for s in suggestions)

    def test_batch_analyze_feedback(self, analyzer):
        """Test batch analysis of multiple feedback messages."""
        feedback_list = [
            "Excellent optimization work!",
            "Security vulnerabilities detected in base image.",
            "Consider using multi-stage builds.",
            "Critical security issues found."
        ]
        
        results = analyzer.batch_analyze_feedback(feedback_list)
        
        assert isinstance(results, dict)
        assert len(results) == len(feedback_list)
        
        for key, optimized_feedback in results.items():
            assert isinstance(optimized_feedback, OptimizedFeedback)
            assert hasattr(optimized_feedback, 'sentiment_analysis')

    def test_generate_sentiment_report(self, analyzer):
        """Test comprehensive sentiment analysis report generation."""
        feedback_list = [
            "Great work on security improvements!",
            "Error: Dockerfile has vulnerabilities.",
            "Consider optimizing layer structure.",
            "Excellent use of multi-stage builds!",
            "Warning: deprecated base image detected."
        ]
        
        report = analyzer.generate_sentiment_report(feedback_list)
        
        assert isinstance(report, dict)
        assert "total_feedback_analyzed" in report
        assert "sentiment_distribution" in report
        assert "average_confidence" in report
        assert "top_emotional_keywords" in report
        assert "recommendations" in report
        
        assert report["total_feedback_analyzed"] == len(feedback_list)
        assert isinstance(report["sentiment_distribution"], dict)
        assert 0.0 <= report["average_confidence"] <= 1.0
        assert isinstance(report["top_emotional_keywords"], list)
        assert isinstance(report["recommendations"], list)

    def test_sentiment_distribution_calculation(self, analyzer):
        """Test that sentiment distribution is calculated correctly."""
        feedback_list = [
            "Excellent security implementation!",  # Very positive
            "Great optimization work!",            # Positive  
            "Standard Docker commands used.",      # Neutral
            "Issues found in configuration.",      # Negative
            "Critical vulnerabilities detected."   # Very negative
        ]
        
        report = analyzer.generate_sentiment_report(feedback_list)
        sentiment_dist = report["sentiment_distribution"]
        
        # Should have representation across sentiment spectrum
        total_sentiments = sum(sentiment_dist.values())
        assert total_sentiments == len(feedback_list)

    def test_keyword_frequency_tracking(self, analyzer):
        """Test that emotional keywords are tracked correctly."""
        feedback_list = [
            "Excellent security work with great optimization!",
            "Security vulnerabilities found, needs fixing.",
            "Great security improvements implemented."
        ]
        
        report = analyzer.generate_sentiment_report(feedback_list)
        keywords = dict(report["top_emotional_keywords"])
        
        # 'security' should appear frequently
        assert 'great' in keywords or 'excellent' in keywords
        # Should track frequency correctly
        assert any(freq >= 2 for freq in keywords.values())

    def test_confidence_scoring(self, analyzer):
        """Test confidence scoring based on keyword density."""
        high_density_text = "Excellent great perfect optimal secure efficient recommended"
        low_density_text = "The Dockerfile contains standard installation commands."
        
        high_analysis = analyzer.analyze_sentiment(high_density_text)
        low_analysis = analyzer.analyze_sentiment(low_density_text)
        
        assert high_analysis.confidence > low_analysis.confidence
        assert high_analysis.confidence <= 0.9  # Max confidence cap
        assert low_analysis.confidence >= 0.5   # Minimum base confidence

    def test_response_template_selection(self, analyzer):
        """Test that appropriate response templates are selected."""
        # Test each sentiment level gets appropriate emoji/tone
        sentiments_to_test = [
            ("Excellent work!", SentimentScore.VERY_POSITIVE, "🎉"),
            ("Good job!", SentimentScore.POSITIVE, "👍"),
            ("Standard approach.", SentimentScore.NEUTRAL, "ℹ️"),
            ("Issues detected.", SentimentScore.NEGATIVE, "⚠️"),
            ("Critical problems found.", SentimentScore.VERY_NEGATIVE, "🚨")
        ]
        
        for text, expected_sentiment, expected_emoji in sentiments_to_test:
            optimized = analyzer.optimize_feedback(text)
            # Template should match sentiment level
            if expected_sentiment in [SentimentScore.VERY_POSITIVE, SentimentScore.POSITIVE]:
                assert any(emoji in optimized.optimized_message 
                          for emoji in ["🎉", "✨", "🚀", "👍", "✅", "🌟"])
            elif expected_sentiment == SentimentScore.NEUTRAL:
                assert any(emoji in optimized.optimized_message 
                          for emoji in ["ℹ️", "📋", "🔍"])
            else:  # Negative sentiments
                assert any(emoji in optimized.optimized_message 
                          for emoji in ["⚠️", "🔧", "💡", "🚨", "🛡️", "⚡"])

    def test_empty_text_handling(self, analyzer):
        """Test handling of empty or minimal text."""
        empty_text = ""
        minimal_text = "OK"
        
        empty_analysis = analyzer.analyze_sentiment(empty_text)
        minimal_analysis = analyzer.analyze_sentiment(minimal_text)
        
        assert empty_analysis.sentiment_score == SentimentScore.NEUTRAL
        assert minimal_analysis.sentiment_score == SentimentScore.NEUTRAL
        assert empty_analysis.confidence >= 0.5
        assert minimal_analysis.confidence >= 0.5

    def test_special_categories_recognition(self, analyzer):
        """Test recognition of special categories like security and performance."""
        security_text = "Security vulnerability found in base image configuration."
        performance_text = "Performance optimization needed for build speed."
        
        security_analysis = analyzer.analyze_sentiment(security_text)
        performance_analysis = analyzer.analyze_sentiment(performance_text)
        
        # Should recognize domain-specific keywords
        assert any('security' in keyword for keyword in security_analysis.emotional_keywords)
        
        # Suggestions should be contextual
        security_suggestions = analyzer._generate_suggestions(
            security_analysis.sentiment_score, 
            security_analysis.emotional_keywords
        )
        assert any('security' in s.lower() for s in security_suggestions)

    @pytest.mark.parametrize("format_type", ["json", "yaml", "text"])
    def test_integration_with_output_formats(self, analyzer, format_type):
        """Test that sentiment analyzer works with different output formats."""
        feedback = "Your Dockerfile has security issues that need attention."
        optimized = analyzer.optimize_feedback(feedback, "security")
        
        # Should work regardless of output format
        assert isinstance(optimized.optimized_message, str)
        assert len(optimized.optimized_message) > 0
        assert optimized.sentiment_analysis.sentiment_score in [
            SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE
        ]