"""
Clinical Data Retrieval Agent
Searches across multiple Clinical Tables APIs based on term analysis
"""

import asyncio
import logging
from typing import Dict, List, Any
from apis.clinical_tables import ClinicalTablesClient

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Agent for retrieving clinical data from multiple sources"""
    
    def __init__(self, client: ClinicalTablesClient):
        self.client = client
    
    async def retrieve(
        self,
        term: str,
        datasets: List[str],
        max_results_per_dataset: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve codes and data from multiple datasets
        
        Args:
            term: Search term
            datasets: List of dataset names to search
            max_results_per_dataset: Max results per dataset
            
        Returns:
            Dictionary with results from all datasets
        """
        logger.info(f"Retrieving data for '{term}' from {len(datasets)} datasets")
        
        results = await self.client.search_multiple(
            term=term,
            datasets=datasets,
            max_results=max_results_per_dataset
        )
        
        # Filter and structure results
        structured_results = {}
        total_matches = 0
        
        for dataset, data in results.items():
            if data.get("count", 0) > 0:
                structured_results[dataset] = {
                    "count": data["count"],
                    "results": self._format_results(dataset, data)
                }
                total_matches += data["count"]
        
        logger.info(f"Retrieved {total_matches} total matches across {len(structured_results)} datasets")
        
        return {
            "term": term,
            "total_matches": total_matches,
            "datasets_searched": len(datasets),
            "datasets_with_results": len(structured_results),
            "results": structured_results
        }
    
    def _format_results(self, dataset: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format results for display"""
        formatted = []
        
        codes = data.get("codes", [])
        raw_data = data.get("data", [])
        
        for i, code in enumerate(codes):
            result = {"code": code}
            
            # Add description if available
            if i < len(raw_data) and raw_data[i]:
                if isinstance(raw_data[i], list):
                    # For datasets like HPO that return [code, description]
                    if len(raw_data[i]) > 1:
                        result["description"] = raw_data[i][1]  # Description is second element
                    elif len(raw_data[i]) > 0:
                        result["description"] = raw_data[i][0]  # Fallback to first element
                elif isinstance(raw_data[i], str):
                    result["description"] = raw_data[i]
            
            result["dataset"] = dataset
            formatted.append(result)
        
        return formatted
    
    async def retrieve_with_alternatives(
        self,
        term: str,
        alternative_terms: List[str],
        datasets: List[str],
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve using the main term and alternative search terms
        
        Args:
            term: Primary search term
            alternative_terms: Alternative terms to try
            datasets: Datasets to search
            max_results: Max results per dataset
            
        Returns:
            Combined results from all search attempts
        """
        all_terms = [term] + alternative_terms
        
        # Search with all terms in parallel
        tasks = [
            self.retrieve(search_term, datasets, max_results)
            for search_term in all_terms
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # Merge results, removing duplicates
        merged_results = self._merge_results(results_list)
        
        return merged_results
    
    def _merge_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple search attempts"""
        if not results_list:
            return {
                "total_matches": 0,
                "datasets_searched": 0,
                "datasets_with_results": 0,
                "results": {}
            }
        
        merged = {
            "total_matches": 0,
            "datasets_searched": results_list[0].get("datasets_searched", 0),
            "datasets_with_results": 0,
            "results": {}
        }
        
        seen_codes = set()
        
        for result_set in results_list:
            for dataset, dataset_results in result_set.get("results", {}).items():
                if dataset not in merged["results"]:
                    merged["results"][dataset] = {
                        "count": 0,
                        "results": []
                    }
                
                # Add unique results
                for item in dataset_results.get("results", []):
                    code = item.get("code")
                    key = f"{dataset}:{code}"
                    
                    if key not in seen_codes:
                        seen_codes.add(key)
                        merged["results"][dataset]["results"].append(item)
                        merged["results"][dataset]["count"] += 1
                        merged["total_matches"] += 1
        
        merged["datasets_with_results"] = len(merged["results"])
        
        return merged
