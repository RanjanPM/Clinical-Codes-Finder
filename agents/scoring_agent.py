"""
Result Scoring Agent
Scores medical code results for relevance to original query
"""

import logging
from typing import Dict, List, Any
from difflib import SequenceMatcher
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import config

logger = logging.getLogger(__name__)


class ResultScoringAgent:
    """Agent for scoring result relevance"""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        self.llm = ChatOpenAI(
            model=model_name or config.llm.SCORING_MODEL,
            temperature=temperature if temperature is not None else config.llm.SCORING_TEMPERATURE
        )
    
    async def score_results(
        self,
        query: str,
        results: Dict[str, Any],
        term_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score all results for relevance to query
        
        Args:
            query: Original user query
            results: Results from retrieval agent
            term_analysis: Analysis from terminology agent
            
        Returns:
            Results dict with relevance_score added to each item
        """
        logger.info(f"Scoring {results.get('total_matches', 0)} results for relevance")
        
        scored_results = results.copy()
        term_type = term_analysis.get("term_type", "unknown")
        primary_datasets = term_analysis.get("primary_datasets", [])
        
        # Score results in each dataset
        for dataset, data in scored_results.get("results", {}).items():
            scored_items = []
            
            # Process in batches to avoid overwhelming the LLM
            items = data.get("results", [])
            
            for item in items:
                score = await self._score_single_result(
                    query=query,
                    code=item.get("code", ""),
                    description=item.get("description", ""),
                    dataset=dataset,
                    term_type=term_type,
                    primary_datasets=primary_datasets
                )
                
                item["relevance_score"] = score
                item["relevance_level"] = self._score_to_level(score)
                scored_items.append(item)
            
            # Sort by relevance
            scored_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            scored_results["results"][dataset]["results"] = scored_items
        
        # Calculate overall quality metrics
        all_scores = []
        for data in scored_results.get("results", {}).values():
            all_scores.extend([
                item.get("relevance_score", 0) 
                for item in data.get("results", [])
            ])
        
        if all_scores:
            scored_results["quality_metrics"] = {
                "avg_relevance": sum(all_scores) / len(all_scores),
                "max_relevance": max(all_scores),
                "min_relevance": min(all_scores),
                "high_quality_count": len([s for s in all_scores if s >= 0.7])
            }
        else:
            scored_results["quality_metrics"] = {
                "avg_relevance": 0.0,
                "max_relevance": 0.0,
                "min_relevance": 0.0,
                "high_quality_count": 0
            }
        
        logger.info(f"Scoring complete. Avg relevance: {scored_results['quality_metrics']['avg_relevance']:.2f}")
        
        return scored_results
    
    async def _score_single_result(
        self,
        query: str,
        code: str,
        description: str,
        dataset: str,
        term_type: str,
        primary_datasets: List[str]
    ) -> float:
        """
        Score a single result using multiple factors
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        
        # Factor 1: Text similarity
        text_sim = self._text_similarity(query.lower(), description.lower())
        score += text_sim * config.scoring.TEXT_SIMILARITY_WEIGHT
        
        # Factor 2: Dataset appropriateness
        dataset_score = config.scoring.DATASET_APPROPRIATENESS_WEIGHT
        if dataset in primary_datasets:
            score += dataset_score
        elif dataset in [d.replace("_", "") for d in primary_datasets]:
            score += dataset_score * 0.75
        else:
            score += dataset_score * 0.25
        
        # Factor 3: Code specificity
        specificity = self._code_specificity(code, dataset)
        score += specificity * config.scoring.CODE_SPECIFICITY_WEIGHT
        
        # Factor 4: Description quality
        desc_quality = self._description_quality(description)
        score += desc_quality * config.scoring.DESCRIPTION_QUALITY_WEIGHT
        
        # Factor 5: Query term presence
        term_presence = self._query_term_presence(query, description)
        score += term_presence * config.scoring.QUERY_TERM_PRESENCE_WEIGHT
        
        return min(1.0, max(0.0, score))
    
    async def _llm_relevance_check(
        self,
        query: str,
        code: str,
        description: str,
        term_type: str
    ) -> float:
        """
        Use LLM to judge relevance (expensive, use sparingly)
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        prompt = f"""Rate the relevance of this medical code to the user's query.

User query: "{query}"
Expected term type: {term_type}

Medical code: {code}
Description: {description}

On a scale of 0.0 to 1.0, how relevant is this code?

Scoring guide:
1.0 = Perfect match (exact intent)
0.8 = Very relevant (directly related)
0.6 = Relevant (related but not exact)
0.4 = Somewhat relevant (tangentially related)
0.2 = Barely relevant (weak connection)
0.0 = Not relevant (unrelated)

Consider:
- Does it match the clinical intent?
- Is it the right type of code?
- Is it too broad or too specific?
- Would a clinician find this useful?

Respond with ONLY a number between 0.0 and 1.0, nothing else."""
        
        try:
            messages = [
                SystemMessage(content="You are a medical coding relevance expert."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            score_str = response.content.strip()
            
            # Extract number from response
            import re
            match = re.search(r'0?\.\d+|1\.0', score_str)
            if match:
                return float(match.group())
            
            # Try to parse as float directly
            return max(0.0, min(1.0, float(score_str)))
            
        except Exception as e:
            logger.warning(f"LLM relevance check failed: {e}")
            return 0.5  # Neutral score on error
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _code_specificity(self, code: str, dataset: str) -> float:
        """
        Estimate code specificity based on structure
        
        More specific codes generally have:
        - More characters/digits
        - More decimal places
        - More qualifiers
        """
        if not code or code == "N/A":
            return 0.3
        
        # ICD codes: more digits = more specific
        if dataset in ["icd10cm", "icd9cm_dx", "icd9cm_sg"]:
            # E11.9 (less specific) vs E11.3211 (more specific)
            parts = code.split(".")
            if len(parts) > 1:
                decimal_digits = len(parts[1])
                return min(1.0, 0.5 + (decimal_digits * 0.15))
            return 0.5
        
        # LOINC codes: presence of system/scale info
        elif dataset == "loinc":
            # Longer LOINC codes tend to be more specific
            return min(1.0, 0.4 + (len(code) * 0.02))
        
        # RxTerms: combination drugs are more specific
        elif dataset in ["rxterms", "drugs"]:
            if "/" in code or "+" in code:
                return 0.8  # Combination drug
            return 0.6
        
        # Default
        return 0.6
    
    def _description_quality(self, description: str) -> float:
        """
        Assess description quality
        
        Higher quality descriptions:
        - Have reasonable length
        - Contain medical terminology
        - Are not generic placeholders
        """
        if not description or description in ["N/A", "No description", "No description available"]:
            return 0.0
        
        desc_lower = description.lower()
        
        # Check for generic/placeholder descriptions
        if len(description) < 10:
            return 0.3
        
        # Good length range
        if 20 <= len(description) <= 200:
            quality = 0.7
        else:
            quality = 0.5
        
        # Bonus for medical terms
        medical_terms = ["blood", "test", "diabetes", "chronic", "acute", "syndrome", 
                        "disease", "disorder", "condition", "procedure", "treatment"]
        if any(term in desc_lower for term in medical_terms):
            quality += 0.2
        
        return min(1.0, quality)
    
    def _query_term_presence(self, query: str, description: str) -> float:
        """
        Check how many query terms appear in description
        
        Returns:
            Ratio of query words found in description (0.0 to 1.0)
        """
        # Normalize and tokenize
        query_lower = query.lower()
        desc_lower = description.lower()
        
        # Remove common stopwords
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "of", "for", "in", "on", "at"}
        query_words = [w for w in query_lower.split() if w not in stopwords and len(w) > 2]
        
        if not query_words:
            return 0.5
        
        # Count matches
        matches = sum(1 for word in query_words if word in desc_lower)
        
        # Also check for partial matches (e.g., "diabetes" in "diabetic")
        partial_matches = sum(1 for word in query_words 
                             if any(word[:4] in desc_word for desc_word in desc_lower.split()))
        
        total_score = (matches + partial_matches * 0.5) / len(query_words)
        
        return min(1.0, total_score)
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to categorical level"""
        if score >= config.scoring.HIGH_RELEVANCE_THRESHOLD:
            return "high"
        elif score >= config.scoring.MEDIUM_RELEVANCE_THRESHOLD:
            return "medium"
        elif score >= config.scoring.LOW_RELEVANCE_THRESHOLD:
            return "low"
        else:
            return "very_low"
    
    async def get_top_results(
        self,
        scored_results: Dict[str, Any],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top N results across all datasets
        
        Args:
            scored_results: Results with relevance scores
            top_n: Number of top results to return
            
        Returns:
            List of top results sorted by relevance
        """
        all_results = []
        
        for dataset, data in scored_results.get("results", {}).items():
            for item in data.get("results", []):
                all_results.append(item)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return all_results[:top_n]
