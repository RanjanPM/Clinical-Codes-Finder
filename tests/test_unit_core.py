"""
Unit Tests for Core Components
Simple, focused tests without complex mocking
"""

import pytest
import time
from datetime import datetime, timedelta
from memory.conversation_memory import ConversationMemory
from config import config


# ============================================================================
# MEMORY TESTS
# ============================================================================

class TestConversationMemory:
    """Unit tests for ConversationMemory"""
    
    @pytest.fixture
    def memory(self):
        """Create fresh memory instance"""
        return ConversationMemory()
    
    def test_initialization(self, memory):
        """Test memory initializes correctly"""
        assert memory.last_query is None
        assert memory.last_results is None
        assert memory.current_page == 0
        assert memory.codes_per_page == 5
    
    def test_query_hash_generation(self, memory):
        """Test query hash is consistent"""
        hash1 = memory._generate_query_hash("diabetes")
        hash2 = memory._generate_query_hash("diabetes")
        hash3 = memory._generate_query_hash("DIABETES")  # Different case
        hash4 = memory._generate_query_hash("hypertension")
        
        # Same query = same hash
        assert hash1 == hash2
        # Case insensitive
        assert hash1 == hash3
        # Different query = different hash
        assert hash1 != hash4
    
    def test_continuation_keyword_detection(self, memory):
        """Test detection of continuation keywords"""
        # Set up previous query
        memory.last_query = "diabetes"
        memory.last_results = {"some": "data"}
        
        # Should detect continuation
        assert memory.is_continuation_request("more") is True
        assert memory.is_continuation_request("next") is True
        assert memory.is_continuation_request("show more") is True
        assert memory.is_continuation_request("continue") is True
        
        # Should NOT detect as continuation
        assert memory.is_continuation_request("hypertension") is False
        assert memory.is_continuation_request("blood pressure") is False
    
    def test_continuation_requires_previous_query(self, memory):
        """Test continuation detection requires previous query"""
        # No previous query
        assert memory.is_continuation_request("more") is False
        
        # Set previous query
        memory.last_query = "diabetes"
        memory.last_results = {"data": "here"}
        
        # Now should detect
        assert memory.is_continuation_request("more") is True
    
    def test_query_caching(self, memory):
        """Test query result caching"""
        query = "diabetes"
        results = {"codes": ["E11.9", "E10.9"], "total": 2}
        
        # Initially not cached
        assert memory.is_cached_query(query) is False
        
        # Cache it
        memory.cache_query_results(query, results)
        
        # Now should be cached
        assert memory.is_cached_query(query) is True
        
        # Retrieve from cache
        cached = memory.get_cached_results(query)
        assert cached is not None
        assert cached["codes"] == results["codes"]
        assert cached["from_cache"] is True
        assert "cache_age_seconds" in cached
    
    def test_cache_case_insensitive(self, memory):
        """Test cache is case insensitive"""
        results = {"codes": ["E11.9"]}
        
        # Cache with one case
        memory.cache_query_results("Diabetes", results)
        
        # Retrieve with different case
        assert memory.is_cached_query("diabetes") is True
        assert memory.is_cached_query("DIABETES") is True
        
        cached = memory.get_cached_results("diabetes")
        assert cached is not None
    
    def test_cache_expiration(self, memory):
        """Test cache respects TTL"""
        query = "diabetes"
        results = {"codes": ["E11.9"]}
        
        # Cache with short TTL
        original_ttl = memory.QUERY_CACHE_TTL
        memory.QUERY_CACHE_TTL = 0.1  # 0.1 seconds
        
        memory.cache_query_results(query, results)
        assert memory.is_cached_query(query) is True
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert memory.is_cached_query(query) is False
        
        # Restore TTL
        memory.QUERY_CACHE_TTL = original_ttl
    
    def test_store_results(self, memory):
        """Test storing query results"""
        query = "diabetes"
        results = {
            "codes_by_system": {
                "ICD-10-CM": [{"code": "E11.9"}, {"code": "E10.9"}]
            }
        }
        
        memory.store_results(query, results, codes_per_page=10)
        
        assert memory.last_query == query
        assert memory.last_results == results
        assert memory.current_page == 0
        assert memory.codes_per_page == 10
    
    def test_clear_cache(self, memory):
        """Test clearing cache"""
        memory.cache_query_results("query1", {"data": 1})
        memory.cache_query_results("query2", {"data": 2})
        
        assert len(memory.query_cache) == 2
        
        memory.clear_cache()
        
        assert len(memory.query_cache) == 0
    
    def test_get_summary(self, memory):
        """Test getting cache summary"""
        memory.cache_query_results("query1", {"data": 1})
        memory.cache_query_results("query2", {"data": 2})
        
        summary = memory.get_summary()
        
        assert "cache_size" in summary
        assert summary["cache_size"] == 2


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Unit tests for Configuration"""
    
    def test_config_exists(self):
        """Test that configuration loads"""
        assert config is not None
    
    def test_agentic_config(self):
        """Test agentic workflow configuration"""
        assert hasattr(config, 'agentic')
        assert config.agentic.MAX_ITERATIONS > 0
        assert 0.0 <= config.agentic.MIN_QUALITY_THRESHOLD <= 1.0
        assert config.agentic.MIN_RESULTS_THRESHOLD >= 0
        assert isinstance(config.agentic.ENABLE_EARLY_STOPPING, bool)
    
    def test_scoring_config(self):
        """Test scoring configuration"""
        assert hasattr(config, 'scoring')
        
        # All weights should be between 0 and 1
        assert 0.0 <= config.scoring.TEXT_SIMILARITY_WEIGHT <= 1.0
        assert 0.0 <= config.scoring.DATASET_APPROPRIATENESS_WEIGHT <= 1.0
        assert 0.0 <= config.scoring.CODE_SPECIFICITY_WEIGHT <= 1.0
        
        # Weights should sum to approximately 1.0
        total = (
            config.scoring.TEXT_SIMILARITY_WEIGHT +
            config.scoring.DATASET_APPROPRIATENESS_WEIGHT +
            config.scoring.CODE_SPECIFICITY_WEIGHT +
            config.scoring.DESCRIPTION_QUALITY_WEIGHT +
            config.scoring.QUERY_TERM_PRESENCE_WEIGHT
        )
        assert 0.99 <= total <= 1.01  # Allow small floating point error
    
    def test_scoring_thresholds(self):
        """Test scoring thresholds are ordered correctly"""
        assert config.scoring.HIGH_RELEVANCE_THRESHOLD > config.scoring.MEDIUM_RELEVANCE_THRESHOLD
        assert config.scoring.MEDIUM_RELEVANCE_THRESHOLD > config.scoring.LOW_RELEVANCE_THRESHOLD
        assert 0.0 <= config.scoring.LOW_RELEVANCE_THRESHOLD <= 1.0
        assert 0.0 <= config.scoring.HIGH_RELEVANCE_THRESHOLD <= 1.0
    
    def test_api_config(self):
        """Test API configuration"""
        assert hasattr(config, 'api')
        assert config.api.BASE_URL.startswith("http")
        assert config.api.RATE_LIMIT > 0
        assert config.api.MAX_RESULTS_PER_DATASET > 0
    
    def test_display_config(self):
        """Test display configuration"""
        assert hasattr(config, 'display')
        assert config.display.MAX_CODES_PER_SYSTEM > 0
        assert isinstance(config.display.SHOW_RELEVANCE_SCORES, bool)
        assert isinstance(config.display.SHOW_SYNTHESIS, bool)
        assert isinstance(config.display.SHOW_QUALITY_METRICS, bool)
    
    def test_llm_config(self):
        """Test LLM configuration"""
        assert hasattr(config, 'llm')
        assert config.llm.TERMINOLOGY_MODEL is not None
        assert config.llm.SYNTHESIS_MODEL is not None
        
        # Temperatures should be between 0 and 1
        assert 0.0 <= config.llm.TERMINOLOGY_TEMPERATURE <= 1.0
        assert 0.0 <= config.llm.SYNTHESIS_TEMPERATURE <= 1.0
    
    def test_quality_weights(self):
        """Test quality calculation weights"""
        assert 0.0 <= config.agentic.QUALITY_RELEVANCE_WEIGHT <= 1.0
        assert 0.0 <= config.agentic.QUALITY_COUNT_WEIGHT <= 1.0
        
        # Should sum to 1.0
        total = config.agentic.QUALITY_RELEVANCE_WEIGHT + config.agentic.QUALITY_COUNT_WEIGHT
        assert 0.99 <= total <= 1.01


# ============================================================================
# API CLIENT TESTS (Simple, no complex mocking)
# ============================================================================

class TestClinicalTablesClient:
    """Simple unit tests for ClinicalTablesClient"""
    
    def test_client_import(self):
        """Test that client can be imported"""
        from apis.clinical_tables import ClinicalTablesClient
        client = ClinicalTablesClient()
        assert client is not None
    
    def test_datasets_configured(self):
        """Test that datasets are configured"""
        from apis.clinical_tables import ClinicalTablesClient
        client = ClinicalTablesClient()
        
        assert hasattr(client, 'DATASETS')
        assert len(client.DATASETS) > 0
        
        # Check expected datasets
        assert "icd10cm" in client.DATASETS
        assert "loinc" in client.DATASETS
        assert "rxterms" in client.DATASETS
        assert "conditions" in client.DATASETS
    



