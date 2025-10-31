"""
Integration Tests for Clinical Term Lookup System
Tests end-to-end workflows and API integrations
"""

import pytest
import asyncio
import os
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.asyncio
class TestClinicalTablesAPIIntegration:
    """Integration tests for Clinical Tables API"""
    
    @pytest.fixture
    async def client(self):
        from apis.clinical_tables import ClinicalTablesClient
        async with ClinicalTablesClient() as client:
            yield client
    
    async def test_icd10cm_search(self, client):
        """Test ICD-10-CM search"""
        result = await client.search("icd10cm", "diabetes", max_results=5)
        
        assert result["count"] > 0
        assert len(result["codes"]) > 0
        assert len(result["data"]) > 0
        assert "E" in result["codes"][0]  # ICD-10-CM codes start with letter
    
    async def test_loinc_search(self, client):
        """Test LOINC search"""
        result = await client.search("loinc", "glucose", max_results=5)
        
        assert result["count"] > 0
        assert len(result["codes"]) > 0
    
    async def test_rxterms_search(self, client):
        """Test RxTerms search"""
        result = await client.search("rxterms", "metformin", max_results=5)
        
        assert result["count"] > 0
        assert len(result["codes"]) > 0
    
    async def test_conditions_search(self, client):
        """Test medical conditions search"""
        result = await client.search("conditions", "hypertension", max_results=5)
        
        assert result["count"] > 0
        assert len(result["codes"]) > 0
    
    async def test_hpo_search(self, client):
        """Test HPO (phenotypes) search"""
        result = await client.search("hpo", "ataxia", max_results=5)
        
        assert result["count"] > 0
        assert len(result["codes"]) > 0
    
    async def test_multi_dataset_search(self, client):
        """Test searching multiple datasets"""
        datasets = ["icd10cm", "conditions", "loinc"]
        results = await client.search_multiple("diabetes", datasets, max_results=3)
        
        assert len(results) > 0
        for dataset in datasets:
            if dataset in results:
                assert "count" in results[dataset]
    
    async def test_cache_functionality(self, client):
        """Test that caching works"""
        import time
        
        # First call - should hit API
        start = time.time()
        result1 = await client.search("icd10cm", "diabetes", max_results=5)
        time1 = time.time() - start
        
        # Second call - should hit cache
        start = time.time()
        result2 = await client.search("icd10cm", "diabetes", max_results=5)
        time2 = time.time() - start
        
        # Cache should be faster
        assert time2 < time1
        assert result1 == result2
    
    async def test_error_handling_invalid_dataset(self, client):
        """Test error handling for invalid dataset"""
        # Should raise ValueError for unknown dataset
        with pytest.raises(ValueError, match="Unknown dataset"):
            await client.search("invalid_dataset", "test", max_results=5)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
class TestFullSystemIntegration:
    """Integration tests for full system with LLM"""
    
    @pytest.fixture
    async def lookup_system(self):
        from main import ClinicalTermLookup
        return ClinicalTermLookup()
    
    async def test_diagnosis_lookup(self, lookup_system):
        """Test lookup of a diagnosis term"""
        result = await lookup_system.lookup("diabetes")
        
        assert result["success"] is True
        assert result["term_type"] == "diagnosis"
        assert "codes_by_system" in result
        assert len(result["codes_by_system"]) > 0
        
        # Should find ICD-10-CM codes
        assert "ICD-10-CM" in result["codes_by_system"] or \
               "ICD10CM" in result["codes_by_system"]
    
    async def test_lab_test_lookup(self, lookup_system):
        """Test lookup of a lab test"""
        result = await lookup_system.lookup("hemoglobin A1c")
        
        assert result["success"] is True
        assert result["term_type"] == "lab_test"
        assert "codes_by_system" in result
    
    async def test_medication_lookup(self, lookup_system):
        """Test lookup of a medication"""
        result = await lookup_system.lookup("metformin 500 mg")
        
        assert result["success"] is True
        assert result["term_type"] == "medication"
        assert "codes_by_system" in result
    
    async def test_phenotype_lookup(self, lookup_system):
        """Test lookup of a phenotype"""
        result = await lookup_system.lookup("ataxia")
        
        assert result["success"] is True
        assert "codes_by_system" in result
    
    async def test_procedure_lookup(self, lookup_system):
        """Test lookup of a medical procedure"""
        result = await lookup_system.lookup("appendectomy")
        
        assert result["success"] is True
        assert "codes_by_system" in result
    
    async def test_workflow_iterations(self, lookup_system):
        """Test that workflow performs iterations"""
        result = await lookup_system.lookup("rare genetic disorder")
        
        assert result["success"] is True
        assert "iteration_count" in result
        assert result["iteration_count"] >= 1
    
    async def test_quality_metrics(self, lookup_system):
        """Test that quality metrics are included"""
        result = await lookup_system.lookup("diabetes")
        
        assert "result_quality" in result
        assert "quality_metrics" in result
        assert "avg_relevance" in result["quality_metrics"]
    
    async def test_synthesis_generation(self, lookup_system):
        """Test that synthesis is generated"""
        result = await lookup_system.lookup("tuberculosis")
        
        assert "synthesis" in result
        assert "executive_summary" in result["synthesis"]
        assert "top_recommendations" in result["synthesis"]
    
    async def test_pagination_info(self, lookup_system):
        """Test that pagination info is included"""
        result = await lookup_system.lookup("hypertension")
        
        assert "pagination_info" in result
        assert "has_more_pages" in result


@pytest.mark.integration
@pytest.mark.asyncio
class TestPaginationIntegration:
    """Integration tests for pagination features"""
    
    @pytest.fixture
    async def lookup_system(self):
        from main import ClinicalTermLookup
        return ClinicalTermLookup()
    
    async def test_pagination_workflow(self, lookup_system):
        """Test complete pagination workflow"""
        # Initial query
        result1 = await lookup_system.lookup("diabetes")
        assert result1["success"] is True
        
        # Check if more pages available
        if result1.get("has_more_pages"):
            # Request more results
            result2 = await lookup_system.lookup("more")
            
            assert result2.get("is_continuation") is True
            assert result2.get("page_number") == 2
            assert "pagination_info" in result2
    
    async def test_continuation_keywords(self, lookup_system):
        """Test various continuation keywords"""
        # Initial query
        await lookup_system.lookup("hypertension")
        
        # Test different keywords
        keywords = ["more", "next", "show more", "continue"]
        
        for keyword in keywords:
            result = await lookup_system.lookup(keyword)
            if result.get("success"):
                assert result.get("is_continuation") is True or \
                       result.get("is_end_of_results") is True
    
    async def test_pagination_range_display(self, lookup_system):
        """Test that pagination ranges are correct"""
        result = await lookup_system.lookup("diabetes")
        
        if result.get("success") and "pagination_info" in result:
            for system, info in result["pagination_info"].items():
                assert "start" in info
                assert "end" in info
                assert "total" in info
                assert info["start"] >= 1
                assert info["end"] >= info["start"]
                assert info["total"] >= info["end"]
    
    async def test_memory_reset_on_new_query(self, lookup_system):
        """Test that memory resets on new query"""
        # First query
        result1 = await lookup_system.lookup("diabetes")
        
        # New query (should reset)
        result2 = await lookup_system.lookup("tuberculosis")
        
        # New queries don't have is_continuation field (only continuations do)
        assert result2.get("is_continuation") != True  # Should be None or False
        assert result2["query"] == "tuberculosis"


@pytest.mark.integration
class TestWorkflowQuality:
    """Integration tests for workflow quality and refinement"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
    async def test_iterative_refinement(self):
        """Test that workflow refines searches when needed"""
        from main import ClinicalTermLookup
        
        lookup = ClinicalTermLookup()
        
        # Use a vague term that might need refinement
        result = await lookup.lookup("blood test")
        
        assert result["success"] is True
        # Should find relevant LOINC codes
        assert len(result.get("codes_by_system", {})) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
    async def test_early_stopping(self):
        """Test that workflow stops early with excellent results"""
        from main import ClinicalTermLookup
        
        lookup = ClinicalTermLookup()
        
        # Use a well-defined term
        result = await lookup.lookup("type 2 diabetes mellitus")
        
        assert result["success"] is True
        # Should achieve good quality quickly
        assert result.get("result_quality", 0) > 0.6


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_api_error_recovery(self):
        """Test recovery from API errors"""
        from apis.clinical_tables import ClinicalTablesClient
        
        async with ClinicalTablesClient() as client:
            # Invalid dataset should raise ValueError
            with pytest.raises(ValueError, match="Unknown dataset"):
                await client.search("nonexistent_dataset", "test")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
    async def test_empty_query_handling(self):
        """Test handling of empty/invalid queries"""
        from main import ClinicalTermLookup
        
        lookup = ClinicalTermLookup()
        
        # Very short query
        result = await lookup.lookup("a")
        # Should still work, even if results are poor
        assert "success" in result
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
    async def test_no_results_handling(self):
        """Test handling when no results found"""
        from main import ClinicalTermLookup
        
        lookup = ClinicalTermLookup()
        
        # Nonsense term unlikely to have results
        result = await lookup.lookup("xyzabc123nonsense")
        
        assert "success" in result
        # Should complete workflow even with no results


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration"""
    
    def test_config_override(self):
        """Test that config can be overridden via environment"""
        import os
        from importlib import reload
        
        # Set env var
        os.environ["MAX_CODES_PER_SYSTEM"] = "15"
        
        # Reload config
        import config
        reload(config)
        
        # Check override worked
        assert config.config.display.MAX_CODES_PER_SYSTEM == 15
        
        # Cleanup
        os.environ.pop("MAX_CODES_PER_SYSTEM", None)
    
    def test_all_datasets_accessible(self):
        """Test that all configured datasets are accessible"""
        from apis.clinical_tables import ClinicalTablesClient
        
        client = ClinicalTablesClient()
        datasets = list(client.DATASETS.keys())
        
        assert len(datasets) > 0
        assert "icd10cm" in datasets
        assert "loinc" in datasets
        assert "rxterms" in datasets


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
