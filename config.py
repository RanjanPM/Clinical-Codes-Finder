"""
Configuration for Clinical Term Lookup System
All configurable parameters in one place
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AgenticConfig:
    """Configuration for agentic workflow behavior"""
    
    # Iteration settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    MIN_RESULTS_THRESHOLD = int(os.getenv("MIN_RESULTS_THRESHOLD", "3"))
    MIN_QUALITY_THRESHOLD = float(os.getenv("MIN_QUALITY_THRESHOLD", "0.6"))
    
    # Quality calculation weights
    QUALITY_RELEVANCE_WEIGHT = float(os.getenv("QUALITY_RELEVANCE_WEIGHT", "0.7"))
    QUALITY_COUNT_WEIGHT = float(os.getenv("QUALITY_COUNT_WEIGHT", "0.3"))
    
    # Early stopping
    ENABLE_EARLY_STOPPING = os.getenv("ENABLE_EARLY_STOPPING", "true").lower() == "true"
    EXCELLENT_QUALITY_THRESHOLD = float(os.getenv("EXCELLENT_QUALITY_THRESHOLD", "0.8"))


class ScoringConfig:
    """Configuration for result scoring"""
    
    # Scoring factor weights (should sum to 1.0)
    TEXT_SIMILARITY_WEIGHT = float(os.getenv("TEXT_SIMILARITY_WEIGHT", "0.30"))
    DATASET_APPROPRIATENESS_WEIGHT = float(os.getenv("DATASET_APPROPRIATENESS_WEIGHT", "0.20"))
    CODE_SPECIFICITY_WEIGHT = float(os.getenv("CODE_SPECIFICITY_WEIGHT", "0.15"))
    DESCRIPTION_QUALITY_WEIGHT = float(os.getenv("DESCRIPTION_QUALITY_WEIGHT", "0.10"))
    QUERY_TERM_PRESENCE_WEIGHT = float(os.getenv("QUERY_TERM_PRESENCE_WEIGHT", "0.25"))
    
    # Relevance thresholds
    HIGH_RELEVANCE_THRESHOLD = float(os.getenv("HIGH_RELEVANCE_THRESHOLD", "0.8"))
    MEDIUM_RELEVANCE_THRESHOLD = float(os.getenv("MEDIUM_RELEVANCE_THRESHOLD", "0.6"))
    LOW_RELEVANCE_THRESHOLD = float(os.getenv("LOW_RELEVANCE_THRESHOLD", "0.4"))
    
    # Enable LLM-based scoring (expensive)
    ENABLE_LLM_SCORING = os.getenv("ENABLE_LLM_SCORING", "false").lower() == "true"


class RefinementConfig:
    """Configuration for search refinement"""
    
    # Refinement strategy triggers
    NO_RESULTS_STRATEGY = os.getenv("NO_RESULTS_STRATEGY", "broaden")  # broaden, alternative
    TOO_MANY_RESULTS_THRESHOLD = int(os.getenv("TOO_MANY_RESULTS_THRESHOLD", "50"))
    TOO_FEW_RESULTS_THRESHOLD = int(os.getenv("TOO_FEW_RESULTS_THRESHOLD", "3"))
    
    # Number of alternative terms to generate
    NUM_ALTERNATIVE_TERMS = int(os.getenv("NUM_ALTERNATIVE_TERMS", "5"))
    
    # Refinement after N iterations
    ALTERNATIVE_STRATEGY_AFTER_ITERATIONS = int(os.getenv("ALTERNATIVE_STRATEGY_AFTER_ITERATIONS", "2"))


class LLMConfig:
    """Configuration for LLM usage"""
    
    # Model selection
    TERMINOLOGY_MODEL = os.getenv("TERMINOLOGY_MODEL", "gpt-4o-mini")
    REFINEMENT_MODEL = os.getenv("REFINEMENT_MODEL", "gpt-4o-mini")
    SCORING_MODEL = os.getenv("SCORING_MODEL", "gpt-4o-mini")
    SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "gpt-4o-mini")
    
    # Temperature settings
    TERMINOLOGY_TEMPERATURE = float(os.getenv("TERMINOLOGY_TEMPERATURE", "0.1"))
    REFINEMENT_TEMPERATURE = float(os.getenv("REFINEMENT_TEMPERATURE", "0.3"))
    SCORING_TEMPERATURE = float(os.getenv("SCORING_TEMPERATURE", "0.1"))
    SYNTHESIS_TEMPERATURE = float(os.getenv("SYNTHESIS_TEMPERATURE", "0.2"))
    
    # Timeout settings (seconds)
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))


class APIConfig:
    """Configuration for Clinical Tables API"""
    
    BASE_URL = os.getenv("CLINICAL_TABLES_BASE_URL", "https://clinicaltables.nlm.nih.gov/api")
    RATE_LIMIT = int(os.getenv("CLINICAL_TABLES_RATE_LIMIT", "100"))
    
    # Results per dataset
    MAX_RESULTS_PER_DATASET = int(os.getenv("MAX_RESULTS_PER_DATASET", "10"))
    
    # Cache settings
    CACHE_TTL_STABLE_CODES = int(os.getenv("CACHE_TTL_STABLE_CODES", "86400"))  # 24 hours
    CACHE_TTL_DYNAMIC_DATA = int(os.getenv("CACHE_TTL_DYNAMIC_DATA", "3600"))   # 1 hour
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("API_RETRY_DELAY", "1.0"))


class DisplayConfig:
    """Configuration for output display"""
    
    # Number of results to display
    MAX_CODES_PER_SYSTEM = int(os.getenv("MAX_CODES_PER_SYSTEM", "5"))
    MAX_TOP_RECOMMENDATIONS = int(os.getenv("MAX_TOP_RECOMMENDATIONS", "3"))
    
    # Display options
    SHOW_RELEVANCE_SCORES = os.getenv("SHOW_RELEVANCE_SCORES", "true").lower() == "true"
    SHOW_ITERATION_HISTORY = os.getenv("SHOW_ITERATION_HISTORY", "true").lower() == "true"
    SHOW_SYNTHESIS = os.getenv("SHOW_SYNTHESIS", "true").lower() == "true"
    SHOW_QUALITY_METRICS = os.getenv("SHOW_QUALITY_METRICS", "true").lower() == "true"
    
    # Verbosity
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR


class Config:
    """Main configuration class"""
    
    agentic = AgenticConfig()
    scoring = ScoringConfig()
    refinement = RefinementConfig()
    llm = LLMConfig()
    api = APIConfig()
    display = DisplayConfig()
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check scoring weights sum to 1.0
        scoring_weights_sum = (
            cls.scoring.TEXT_SIMILARITY_WEIGHT +
            cls.scoring.DATASET_APPROPRIATENESS_WEIGHT +
            cls.scoring.CODE_SPECIFICITY_WEIGHT +
            cls.scoring.DESCRIPTION_QUALITY_WEIGHT +
            cls.scoring.QUERY_TERM_PRESENCE_WEIGHT
        )
        
        if abs(scoring_weights_sum - 1.0) > 0.01:
            issues.append(
                f"Scoring weights sum to {scoring_weights_sum:.2f}, should be 1.0"
            )
        
        # Check quality weights sum to 1.0
        quality_weights_sum = (
            cls.agentic.QUALITY_RELEVANCE_WEIGHT +
            cls.agentic.QUALITY_COUNT_WEIGHT
        )
        
        if abs(quality_weights_sum - 1.0) > 0.01:
            issues.append(
                f"Quality weights sum to {quality_weights_sum:.2f}, should be 1.0"
            )
        
        # Check thresholds are in valid range
        if not 0 <= cls.agentic.MIN_QUALITY_THRESHOLD <= 1:
            issues.append("MIN_QUALITY_THRESHOLD must be between 0 and 1")
        
        if cls.agentic.MAX_ITERATIONS < 1:
            issues.append("MAX_ITERATIONS must be at least 1")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "agentic": {
                "max_iterations": cls.agentic.MAX_ITERATIONS,
                "min_results_threshold": cls.agentic.MIN_RESULTS_THRESHOLD,
                "min_quality_threshold": cls.agentic.MIN_QUALITY_THRESHOLD,
                "enable_early_stopping": cls.agentic.ENABLE_EARLY_STOPPING,
            },
            "scoring": {
                "weights": {
                    "text_similarity": cls.scoring.TEXT_SIMILARITY_WEIGHT,
                    "dataset_appropriateness": cls.scoring.DATASET_APPROPRIATENESS_WEIGHT,
                    "code_specificity": cls.scoring.CODE_SPECIFICITY_WEIGHT,
                    "description_quality": cls.scoring.DESCRIPTION_QUALITY_WEIGHT,
                    "query_term_presence": cls.scoring.QUERY_TERM_PRESENCE_WEIGHT,
                },
                "thresholds": {
                    "high": cls.scoring.HIGH_RELEVANCE_THRESHOLD,
                    "medium": cls.scoring.MEDIUM_RELEVANCE_THRESHOLD,
                    "low": cls.scoring.LOW_RELEVANCE_THRESHOLD,
                }
            },
            "refinement": {
                "too_many_results_threshold": cls.refinement.TOO_MANY_RESULTS_THRESHOLD,
                "num_alternative_terms": cls.refinement.NUM_ALTERNATIVE_TERMS,
            },
            "llm": {
                "models": {
                    "terminology": cls.llm.TERMINOLOGY_MODEL,
                    "refinement": cls.llm.REFINEMENT_MODEL,
                    "scoring": cls.llm.SCORING_MODEL,
                    "synthesis": cls.llm.SYNTHESIS_MODEL,
                },
            },
            "api": {
                "base_url": cls.api.BASE_URL,
                "max_results_per_dataset": cls.api.MAX_RESULTS_PER_DATASET,
            },
            "display": {
                "max_codes_per_system": cls.display.MAX_CODES_PER_SYSTEM,
                "show_relevance_scores": cls.display.SHOW_RELEVANCE_SCORES,
            }
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        import json
        print("=" * 80)
        print("CLINICAL TERM LOOKUP SYSTEM - CONFIGURATION")
        print("=" * 80)
        print(json.dumps(cls.to_dict(), indent=2))
        print("=" * 80)


# Singleton instance
config = Config()
