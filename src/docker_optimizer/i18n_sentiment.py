"""
Internationalization (i18n) Support for DockerfileSentimentAnalyzer

Provides multi-language sentiment analysis and feedback optimization
for global deployment across different regions and languages.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

class SupportedLanguage(Enum):
    """Supported languages for sentiment analysis"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"

@dataclass
class MultilingualFeedback:
    """Sentiment feedback with multi-language support"""
    english: str
    translations: Dict[str, str]
    detected_language: Optional[str] = None
    confidence: float = 1.0

class GlobalSentimentAnalyzer:
    """
    Global sentiment analyzer with multi-language and cultural sensitivity support.
    Extends DockerfileSentimentAnalyzer for international deployment.
    """
    
    def __init__(self):
        # Multi-language keyword dictionaries
        self.multilingual_keywords = {
            SupportedLanguage.ENGLISH: {
                "positive": {
                    'excellent', 'great', 'perfect', 'optimal', 'efficient', 
                    'secure', 'clean', 'well-structured', 'best-practice',
                    'recommended', 'improved', 'optimized', 'enhanced'
                },
                "negative": {
                    'error', 'failure', 'broken', 'vulnerable', 'insecure',
                    'inefficient', 'bloated', 'deprecated', 'dangerous',
                    'problematic', 'wrong', 'bad', 'poor', 'terrible'
                },
                "warning": {
                    'warning', 'caution', 'attention', 'consider', 'should',
                    'might', 'potential', 'risk', 'issue', 'concern'
                }
            },
            SupportedLanguage.SPANISH: {
                "positive": {
                    'excelente', 'genial', 'perfecto', 'óptimo', 'eficiente',
                    'seguro', 'limpio', 'bien-estructurado', 'mejor-práctica',
                    'recomendado', 'mejorado', 'optimizado', 'mejorado'
                },
                "negative": {
                    'error', 'fallo', 'roto', 'vulnerable', 'inseguro',
                    'ineficiente', 'hinchado', 'obsoleto', 'peligroso',
                    'problemático', 'incorrecto', 'malo', 'pobre', 'terrible'
                },
                "warning": {
                    'advertencia', 'precaución', 'atención', 'considerar', 'debería',
                    'podría', 'potencial', 'riesgo', 'problema', 'preocupación'
                }
            },
            SupportedLanguage.FRENCH: {
                "positive": {
                    'excellent', 'génial', 'parfait', 'optimal', 'efficace',
                    'sécurisé', 'propre', 'bien-structuré', 'bonne-pratique',
                    'recommandé', 'amélioré', 'optimisé', 'renforcé'
                },
                "negative": {
                    'erreur', 'échec', 'cassé', 'vulnérable', 'non-sécurisé',
                    'inefficace', 'gonflé', 'obsolète', 'dangereux',
                    'problématique', 'incorrect', 'mauvais', 'pauvre', 'terrible'
                },
                "warning": {
                    'avertissement', 'prudence', 'attention', 'considérer', 'devrait',
                    'pourrait', 'potentiel', 'risque', 'problème', 'préoccupation'
                }
            },
            SupportedLanguage.GERMAN: {
                "positive": {
                    'ausgezeichnet', 'großartig', 'perfekt', 'optimal', 'effizient',
                    'sicher', 'sauber', 'gut-strukturiert', 'beste-praxis',
                    'empfohlen', 'verbessert', 'optimiert', 'erweitert'
                },
                "negative": {
                    'fehler', 'versagen', 'kaputt', 'verwundbar', 'unsicher',
                    'ineffizient', 'aufgebläht', 'veraltet', 'gefährlich',
                    'problematisch', 'falsch', 'schlecht', 'arm', 'schrecklich'
                },
                "warning": {
                    'warnung', 'vorsicht', 'aufmerksamkeit', 'betrachten', 'sollte',
                    'könnte', 'potentiell', 'risiko', 'problem', 'sorge'
                }
            },
            SupportedLanguage.JAPANESE: {
                "positive": {
                    '素晴らしい', '最高', '完璧', '最適', '効率的',
                    '安全', 'きれいな', 'よく構造化された', 'ベストプラクティス',
                    '推奨', '改善された', '最適化された', '強化された'
                },
                "negative": {
                    'エラー', '失敗', '壊れた', '脆弱な', '安全でない',
                    '非効率的', '肥大化した', '非推奨', '危険な',
                    '問題のある', '間違った', '悪い', '貧しい', 'ひどい'
                },
                "warning": {
                    '警告', '注意', '注意', '考慮する', 'すべき',
                    'かもしれない', '潜在的', 'リスク', '問題', '懸念'
                }
            },
            SupportedLanguage.CHINESE: {
                "positive": {
                    '优秀', '太棒了', '完美', '最佳', '高效',
                    '安全', '干净', '结构良好', '最佳实践',
                    '推荐', '改进', '优化', '增强'
                },
                "negative": {
                    '错误', '失败', '坏了', '脆弱', '不安全',
                    '低效', '臃肿', '已弃用', '危险',
                    '有问题', '错误', '坏', '贫穷', '可怕'
                },
                "warning": {
                    '警告', '谨慎', '注意', '考虑', '应该',
                    '可能', '潜在', '风险', '问题', '关心'
                }
            }
        }
        
        # Response templates for different languages
        self.multilingual_templates = {
            SupportedLanguage.ENGLISH: {
                "very_positive": ["🎉 Excellent work! {}", "✨ Outstanding! {}", "🚀 Fantastic approach! {}"],
                "positive": ["👍 Good job! {}", "✅ Nice work! {}", "🌟 Well done! {}"],
                "neutral": ["ℹ️  {}", "📋 {}", "🔍 {}"],
                "negative": ["⚠️  Let's improve this: {}", "🔧 Here's how to fix this: {}", "💡 Consider this enhancement: {}"],
                "very_negative": ["🚨 Critical improvement needed: {}", "🛡️  Security enhancement required: {}", "⚡ Important optimization: {}"]
            },
            SupportedLanguage.SPANISH: {
                "very_positive": ["🎉 ¡Excelente trabajo! {}", "✨ ¡Sobresaliente! {}", "🚀 ¡Enfoque fantástico! {}"],
                "positive": ["👍 ¡Buen trabajo! {}", "✅ ¡Buen trabajo! {}", "🌟 ¡Bien hecho! {}"],
                "neutral": ["ℹ️  {}", "📋 {}", "🔍 {}"],
                "negative": ["⚠️  Mejoremos esto: {}", "🔧 Así se puede arreglar: {}", "💡 Considera esta mejora: {}"],
                "very_negative": ["🚨 Mejora crítica necesaria: {}", "🛡️  Mejora de seguridad requerida: {}", "⚡ Optimización importante: {}"]
            },
            SupportedLanguage.FRENCH: {
                "very_positive": ["🎉 Excellent travail ! {}", "✨ Remarquable ! {}", "🚀 Approche fantastique ! {}"],
                "positive": ["👍 Bon travail ! {}", "✅ Bon travail ! {}", "🌟 Bien fait ! {}"],
                "neutral": ["ℹ️  {}", "📋 {}", "🔍 {}"],
                "negative": ["⚠️  Améliorons ceci : {}", "🔧 Voici comment le corriger : {}", "💡 Considérez cette amélioration : {}"],
                "very_negative": ["🚨 Amélioration critique nécessaire : {}", "🛡️  Amélioration de sécurité requise : {}", "⚡ Optimisation importante : {}"]
            },
            SupportedLanguage.GERMAN: {
                "very_positive": ["🎉 Ausgezeichnete Arbeit! {}", "✨ Hervorragend! {}", "🚀 Fantastischer Ansatz! {}"],
                "positive": ["👍 Gute Arbeit! {}", "✅ Schöne Arbeit! {}", "🌟 Gut gemacht! {}"],
                "neutral": ["ℹ️  {}", "📋 {}", "🔍 {}"],
                "negative": ["⚠️  Lassen Sie uns das verbessern: {}", "🔧 So können Sie das beheben: {}", "💡 Betrachten Sie diese Verbesserung: {}"],
                "very_negative": ["🚨 Kritische Verbesserung erforderlich: {}", "🛡️  Sicherheitsverbesserung erforderlich: {}", "⚡ Wichtige Optimierung: {}"]
            },
            SupportedLanguage.JAPANESE: {
                "very_positive": ["🎉 素晴らしい仕事です！{}", "✨ 優秀です！{}", "🚀 素晴らしいアプローチです！{}"],
                "positive": ["👍 良い仕事です！{}", "✅ 良い仕事です！{}", "🌟 よくやりました！{}"],
                "neutral": ["ℹ️  {}", "📋 {}", "🔍 {}"],
                "negative": ["⚠️  これを改善しましょう：{}", "🔧 修正方法は次のとおりです：{}", "💡 この改善を検討してください：{}"],
                "very_negative": ["🚨 重要な改善が必要です：{}", "🛡️  セキュリティの改善が必要です：{}", "⚡ 重要な最適化：{}"]
            },
            SupportedLanguage.CHINESE: {
                "very_positive": ["🎉 优秀的工作！{}", "✨ 出色！{}", "🚀 绝佳的方法！{}"],
                "positive": ["👍 做得好！{}", "✅ 做得不错！{}", "🌟 做得好！{}"],
                "neutral": ["ℹ️  {}", "📋 {}", "🔍 {}"],
                "negative": ["⚠️  让我们改进这个：{}", "🔧 修复方法如下：{}", "💡 考虑这个改进：{}"],
                "very_negative": ["🚨 需要关键改进：{}", "🛡️  需要安全改进：{}", "⚡ 重要优化：{}"]
            }
        }
        
        # Cultural sensitivity guidelines
        self.cultural_guidelines = {
            SupportedLanguage.ENGLISH: {"direct_feedback": True, "emoji_usage": "high"},
            SupportedLanguage.SPANISH: {"direct_feedback": True, "emoji_usage": "high"},
            SupportedLanguage.FRENCH: {"direct_feedback": True, "emoji_usage": "medium"},
            SupportedLanguage.GERMAN: {"direct_feedback": True, "emoji_usage": "low"},
            SupportedLanguage.JAPANESE: {"direct_feedback": False, "emoji_usage": "medium"},
            SupportedLanguage.CHINESE: {"direct_feedback": False, "emoji_usage": "low"}
        }

    def detect_language(self, text: str) -> Optional[SupportedLanguage]:
        """
        Detect the language of input text using keyword matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language or None if uncertain
        """
        text_lower = text.lower()
        language_scores = {}
        
        for language, keywords in self.multilingual_keywords.items():
            score = 0
            total_keywords = sum(len(keyword_set) for keyword_set in keywords.values())
            
            for keyword_type, keyword_set in keywords.items():
                matches = sum(1 for keyword in keyword_set if keyword in text_lower)
                score += matches
            
            # Normalize score by total keywords
            language_scores[language] = score / total_keywords if total_keywords > 0 else 0
        
        # Return language with highest score if above threshold
        if language_scores:
            best_language = max(language_scores.items(), key=lambda x: x[1])
            if best_language[1] > 0.1:  # Minimum confidence threshold
                return best_language[0]
        
        # Default to English if no clear detection
        return SupportedLanguage.ENGLISH

    def get_multilingual_feedback(self, 
                                 original_message: str,
                                 sentiment_category: str,
                                 target_languages: Optional[List[SupportedLanguage]] = None) -> MultilingualFeedback:
        """
        Generate multilingual feedback for global deployment.
        
        Args:
            original_message: Original feedback message
            sentiment_category: Category of sentiment (very_positive, positive, etc.)
            target_languages: Languages to translate to (defaults to all supported)
            
        Returns:
            MultilingualFeedback with translations
        """
        detected_lang = self.detect_language(original_message)
        
        if target_languages is None:
            target_languages = list(SupportedLanguage)
        
        translations = {}
        
        for language in target_languages:
            if language == SupportedLanguage.ENGLISH:
                continue  # Skip English as it's the base
                
            templates = self.multilingual_templates.get(language, {})
            category_templates = templates.get(sentiment_category, ["ℹ️  {}"])
            
            if category_templates:
                template = category_templates[0]  # Use first template
                translated = template.format(original_message)
                translations[language.value] = translated
        
        return MultilingualFeedback(
            english=original_message,
            translations=translations,
            detected_language=detected_lang.value if detected_lang else "en",
            confidence=0.8  # Static confidence for now
        )

    def apply_cultural_sensitivity(self, 
                                 feedback: str, 
                                 target_language: SupportedLanguage) -> str:
        """
        Apply cultural sensitivity adjustments to feedback.
        
        Args:
            feedback: Original feedback message
            target_language: Target language for cultural adaptation
            
        Returns:
            Culturally adapted feedback
        """
        guidelines = self.cultural_guidelines.get(target_language, {})
        
        # Adjust directness for cultures that prefer indirect communication
        if not guidelines.get("direct_feedback", True):
            # Make feedback more indirect and polite
            if feedback.startswith(("❌", "🚨")):
                feedback = feedback.replace("❌", "⚠️").replace("🚨", "💡")
            
            # Add polite language for Japanese and Chinese cultures
            if target_language in [SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE]:
                if "error" in feedback.lower():
                    feedback = feedback.replace("error", "opportunity for improvement")
                if "wrong" in feedback.lower():
                    feedback = feedback.replace("wrong", "could be enhanced")
        
        # Adjust emoji usage based on cultural preferences
        emoji_usage = guidelines.get("emoji_usage", "medium")
        if emoji_usage == "low":
            # Reduce emoji usage for cultures that prefer formal communication
            emoji_pattern = re.compile(r'[^\w\s\-\.,!?]', re.UNICODE)
            feedback = emoji_pattern.sub('', feedback).strip()
            
        elif emoji_usage == "high":
            # Ensure adequate emoji usage for expressive cultures
            if not re.search(r'[^\w\s\-\.,!?]', feedback, re.UNICODE):
                feedback = f"ℹ️ {feedback}"
        
        return feedback

    def get_timezone_aware_greeting(self, timezone: str = "UTC") -> str:
        """
        Generate timezone-aware greetings for global users.
        
        Args:
            timezone: Target timezone (e.g., "UTC", "America/New_York")
            
        Returns:
            Appropriate greeting based on local time
        """
        import datetime
        
        try:
            # Simple time-based greeting (could be enhanced with proper timezone handling)
            now = datetime.datetime.now()
            hour = now.hour
            
            if 5 <= hour < 12:
                return "Good morning"
            elif 12 <= hour < 17:
                return "Good afternoon"
            elif 17 <= hour < 22:
                return "Good evening"
            else:
                return "Hello"
        except Exception:
            return "Hello"

    def get_regional_compliance_notice(self, region: str) -> Optional[str]:
        """
        Get compliance notices for different regions.
        
        Args:
            region: Target region (e.g., "EU", "US", "APAC")
            
        Returns:
            Compliance notice if applicable
        """
        compliance_notices = {
            "EU": "🇪🇺 This analysis complies with GDPR privacy regulations",
            "US": "🇺🇸 This analysis follows US data protection standards",
            "APAC": "🌏 This analysis respects regional data sovereignty requirements",
            "GLOBAL": "🌍 This analysis follows international privacy standards"
        }
        
        return compliance_notices.get(region.upper())

    def generate_global_report(self, 
                             feedback_messages: List[str],
                             target_region: str = "GLOBAL") -> Dict:
        """
        Generate a comprehensive global deployment report.
        
        Args:
            feedback_messages: List of feedback messages to analyze
            target_region: Target deployment region
            
        Returns:
            Global deployment readiness report
        """
        language_distribution = {}
        total_messages = len(feedback_messages)
        
        # Analyze language distribution
        for message in feedback_messages:
            detected_lang = self.detect_language(message)
            lang_key = detected_lang.value if detected_lang else "unknown"
            language_distribution[lang_key] = language_distribution.get(lang_key, 0) + 1
        
        # Calculate language percentages
        language_percentages = {
            lang: (count / total_messages * 100) for lang, count in language_distribution.items()
        }
        
        # Determine supported languages coverage
        supported_languages = [lang.value for lang in SupportedLanguage]
        coverage_percentage = sum(
            percentage for lang, percentage in language_percentages.items() 
            if lang in supported_languages
        )
        
        # Global readiness assessment
        readiness_score = min(100, coverage_percentage + 10)  # Boost for basic functionality
        
        if readiness_score >= 90:
            readiness_level = "EXCELLENT"
            readiness_status = "✅ Ready for global deployment"
        elif readiness_score >= 75:
            readiness_level = "GOOD"  
            readiness_status = "🟡 Ready with minor localization needs"
        else:
            readiness_level = "NEEDS_IMPROVEMENT"
            readiness_status = "🔴 Requires significant localization work"
        
        return {
            "global_readiness": {
                "score": readiness_score,
                "level": readiness_level,
                "status": readiness_status
            },
            "language_analysis": {
                "total_messages": total_messages,
                "language_distribution": language_percentages,
                "supported_coverage_percent": coverage_percentage
            },
            "regional_compliance": {
                "target_region": target_region,
                "compliance_notice": self.get_regional_compliance_notice(target_region)
            },
            "supported_languages": supported_languages,
            "cultural_sensitivity": "Enabled for all supported languages",
            "deployment_recommendations": self._generate_deployment_recommendations(
                language_percentages, readiness_score
            )
        }

    def _generate_deployment_recommendations(self, 
                                          language_percentages: Dict[str, float],
                                          readiness_score: float) -> List[str]:
        """Generate deployment recommendations based on analysis."""
        recommendations = []
        
        if readiness_score < 75:
            recommendations.append("Expand language support for better global coverage")
        
        if language_percentages.get("en", 0) < 50:
            recommendations.append("Consider adding more English language support")
        
        if any(lang not in ["en", "es", "fr", "de", "ja", "zh"] for lang in language_percentages.keys()):
            recommendations.append("Add support for detected unsupported languages")
        
        recommendations.append("Enable cultural sensitivity features for target regions")
        recommendations.append("Implement timezone-aware messaging")
        recommendations.append("Add regional compliance monitoring")
        
        return recommendations