"""
Clinical Tables API Client
Handles all interactions with the Clinical Tables API endpoints
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
import requests
from cachetools import TTLCache
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)


class ClinicalTablesClient:
    """Client for interacting with Clinical Tables API"""
    
    BASE_URL = config.api.BASE_URL
    
    # Dataset endpoints for different coding systems
    DATASETS = {
        "icd10cm": "icd10cm/v3/search",
        "icd11": "icd11/v3/search",
        "icd9cm_dx": "icd9cm_dx/v3/search",
        "icd9cm_sg": "icd9cm_sg/v3/search",
        "loinc": "loinc_items/v3/search",
        "rxterms": "rxterms/v3/search",
        "hcpcs": "hcpcs/v3/search",
        "ucum": "ucum/v3/search",
        "hpo": "hpo/v3/search",
        "npi_idv": "npi_idv/v3/search",
        "npi_org": "npi_org/v3/search",
        "conditions": "conditions/v3/search",
        "procedures": "procedures/v3/search",
        "drugs": "rxterms/v3/search",
        "clinvar": "clinvar/v3/search",
        "genes": "genes/v3/search",
        "snps": "snps/v3/search",
        "genetic_diseases": "disease_names/v3/search",
        "pharmvar": "pharmvar_star_alleles/v3/search"
    }
    
    def __init__(self, rate_limit: int = None):
        self.rate_limit = rate_limit or config.api.RATE_LIMIT
        self.session: Optional[aiohttp.ClientSession] = None
        # Cache with configurable TTL for stable medical codes
        self.cache = TTLCache(
            maxsize=config.api.CACHE_MAX_SIZE, 
            ttl=config.api.CACHE_TTL_STABLE_CODES
        )
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, dataset: str, term: str, max_results: int) -> str:
        """Generate cache key for a query"""
        return f"{dataset}:{term}:{max_results}"
    
    async def search(
        self, 
        dataset: str, 
        term: str, 
        max_results: int = None,
        df: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search a Clinical Tables dataset
        
        Args:
            dataset: Dataset name (e.g., 'icd10cm', 'loinc')
            term: Search term
            max_results: Maximum number of results
            df: Display fields to return
            
        Returns:
            Dictionary with search results
        """
        cache_key = self._get_cache_key(dataset, term, max_results)
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Cache hit for {dataset}: {term}")
            return self.cache[cache_key]
        
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(self.DATASETS.keys())}")
        
        endpoint = self.DATASETS[dataset]
        url = f"{self.BASE_URL}/{endpoint}"
        
        max_results = max_results or config.api.MAX_RESULTS_PER_DATASET
        params = {
            "terms": term,
            "maxList": max_results
        }
        
        # Add search fields for datasets that support them
        if dataset in ["icd10cm", "icd11", "icd9cm_dx", "icd9cm_sg"]:
            params["sf"] = "code,name"
        
        if df:
            params["df"] = df
        
        try:
            if not self.session:
                # Fallback to synchronous request if no session
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                result = response.json()
            else:
                async with self.session.get(url, params=params, timeout=10) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            # Parse response format: [count, [codes], null, [data]]
            if len(result) >= 4:
                parsed_result = {
                    "count": result[0],
                    "codes": result[1] if result[1] else [],
                    "data": result[3] if result[3] else []
                }
            else:
                parsed_result = {"count": 0, "codes": [], "data": []}
            
            # Cache the result
            self.cache[cache_key] = parsed_result
            logger.info(f"API call successful for {dataset}: {term} - Found {parsed_result['count']} results")
            
            return parsed_result
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error searching {dataset} for '{term}': {e}")
            return {"count": 0, "codes": [], "data": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Error searching {dataset} for '{term}': {e}")
            return {"count": 0, "codes": [], "data": [], "error": str(e)}
    
    async def search_multiple(
        self, 
        term: str, 
        datasets: List[str], 
        max_results: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Search multiple datasets in parallel
        
        Args:
            term: Search term
            datasets: List of dataset names
            max_results: Maximum results per dataset
            
        Returns:
            Dictionary mapping dataset names to results
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        tasks = [
            self.search(dataset, term, max_results)
            for dataset in datasets
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            dataset: result if not isinstance(result, Exception) else {"count": 0, "codes": [], "data": [], "error": str(result)}
            for dataset, result in zip(datasets, results)
        }
    
    def search_sync(
        self, 
        dataset: str, 
        term: str, 
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Synchronous version of search for non-async contexts"""
        cache_key = self._get_cache_key(dataset, term, max_results)
        
        if cache_key in self.cache:
            logger.info(f"Cache hit for {dataset}: {term}")
            return self.cache[cache_key]
        
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        endpoint = self.DATASETS[dataset]
        url = f"{self.BASE_URL}/{endpoint}"
        
        params = {"terms": term, "maxList": max_results}
        if dataset in ["icd10cm", "icd11", "icd9cm_dx", "icd9cm_sg"]:
            params["sf"] = "code,name"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            parsed_result = {
                "count": result[0],
                "codes": result[1] if len(result) > 1 and result[1] else [],
                "data": result[3] if len(result) > 3 and result[3] else []
            }
            
            self.cache[cache_key] = parsed_result
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in sync search: {e}")
            return {"count": 0, "codes": [], "data": [], "error": str(e)}
