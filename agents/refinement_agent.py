"""
Search Refinement Agent
Dynamically adjusts search strategy based on result quality
"""

import logging
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import config

logger = logging.getLogger(__name__)


class SearchRefinementAgent:
    """Agent for refining search strategies based on results"""
    
    SYSTEM_PROMPT = """You are a medical search strategy expert. Your job is to analyze search results and suggest refinements to improve result quality.

When results are insufficient, suggest:
- Broader terms (synonyms, parent categories, common abbreviations)
- Alternative medical terminology
- Different clinical perspectives

When results are too numerous, suggest:
- More specific terms
- Additional qualifiers (type, severity, location)
- Focused subsets

Respond in JSON format with:
{
    "strategy": "broaden|narrow|alternative|sufficient",
    "new_search_terms": ["term1", "term2", "term3"],
    "reasoning": "explanation of the refinement",
    "confidence": 0.0-1.0
}"""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        self.llm = ChatOpenAI(
            model=model_name or config.llm.REFINEMENT_MODEL,
            temperature=temperature if temperature is not None else config.llm.REFINEMENT_TEMPERATURE
        )
    
    async def refine_strategy(
        self,
        original_query: str,
        term_type: str,
        previous_results: Dict[str, Any],
        iteration: int,
        search_history: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze results and suggest search refinement
        
        Args:
            original_query: Original user query
            term_type: Type of medical term (diagnosis, lab_test, etc.)
            previous_results: Results from previous search
            iteration: Current iteration number
            search_history: Previous search terms tried
            
        Returns:
            Dictionary with refinement strategy
        """
        total_matches = previous_results.get("total_matches", 0)
        datasets_with_results = previous_results.get("datasets_with_results", 0)
        
        logger.info(f"Refining search strategy: {total_matches} matches, iteration {iteration}")
        
        # Determine refinement strategy
        if total_matches < config.refinement.TOO_FEW_RESULTS_THRESHOLD:
            return await self._broaden_search(
                original_query, term_type, search_history
            )
        elif total_matches > config.refinement.TOO_MANY_RESULTS_THRESHOLD:
            return await self._narrow_search(
                original_query, term_type, previous_results, search_history
            )
        elif iteration >= config.refinement.ALTERNATIVE_STRATEGY_AFTER_ITERATIONS:
            return await self._alternative_approach(
                original_query, term_type, search_history
            )
        else:
            # Results are acceptable
            return {
                "strategy": "sufficient",
                "new_search_terms": [],
                "reasoning": "Result quality is acceptable",
                "confidence": 0.9
            }
    
    async def _broaden_search(
        self,
        query: str,
        term_type: str,
        search_history: List[str]
    ) -> Dict[str, Any]:
        """Generate broader search terms when no results found"""
        
        prompt = f"""No results found for the medical term: "{query}"
Term type: {term_type}
Already tried: {', '.join(search_history) if search_history else 'None'}

Suggest 3-5 BROADER medical search terms that might retrieve results.

Strategies:
- Use medical synonyms (e.g., "MI" for "myocardial infarction")
- Try common abbreviations (e.g., "HTN" for "hypertension")
- Use parent categories (e.g., "diabetes" for "type 2 diabetes")
- Try simpler lay terms (e.g., "heart attack" for "acute coronary syndrome")
- Consider related conditions or tests

IMPORTANT: Avoid terms already in the search history.
Respond ONLY with valid JSON, no additional text."""
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            result = self._parse_json_response(response.content)
            
            result["strategy"] = "broaden"
            logger.info(f"Broadening search with terms: {result.get('new_search_terms')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in broaden_search: {e}")
            # Fallback to simple broadening
            return self._fallback_broaden(query, search_history)
    
    async def _narrow_search(
        self,
        query: str,
        term_type: str,
        previous_results: Dict[str, Any],
        search_history: List[str]
    ) -> Dict[str, Any]:
        """Generate more specific terms when too many results"""
        
        # Analyze top results to understand common patterns
        sample_results = self._get_sample_results(previous_results, limit=5)
        
        prompt = f"""Too many results ({previous_results.get('total_matches', 0)}) found for: "{query}"
Term type: {term_type}
Already tried: {', '.join(search_history) if search_history else 'None'}

Sample of current results:
{sample_results}

Suggest 3-5 MORE SPECIFIC medical search terms to narrow the results.

Strategies:
- Add qualifiers (acute vs chronic, primary vs secondary)
- Specify type or subtype (Type 1 vs Type 2 diabetes)
- Add anatomical location (left, right, upper, lower)
- Specify severity (mild, moderate, severe)
- Add temporal aspects (new onset, recurrent, chronic)

IMPORTANT: Avoid terms already in the search history.
Respond ONLY with valid JSON, no additional text."""
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            result = self._parse_json_response(response.content)
            
            result["strategy"] = "narrow"
            logger.info(f"Narrowing search with terms: {result.get('new_search_terms')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in narrow_search: {e}")
            return self._fallback_narrow(query, search_history)
    
    async def _alternative_approach(
        self,
        query: str,
        term_type: str,
        search_history: List[str]
    ) -> Dict[str, Any]:
        """Try completely different search approach after multiple iterations"""
        
        prompt = f"""After {len(search_history)} attempts, still not finding optimal results for: "{query}"
Term type: {term_type}
Previously tried: {', '.join(search_history)}

Suggest {config.refinement.NUM_ALTERNATIVE_TERMS} ALTERNATIVE medical search approaches that take a completely different angle.

Strategies:
- Try related symptoms or presentations
- Use procedure or treatment names instead of conditions
- Search by etiology or cause
- Use clinical presentation terms
- Try patient-friendly terminology
- Consider differential diagnoses

Be creative and think outside the box!
Respond ONLY with valid JSON, no additional text."""
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            result = self._parse_json_response(response.content)
            
            result["strategy"] = "alternative"
            logger.info(f"Alternative approach with terms: {result.get('new_search_terms')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in alternative_approach: {e}")
            return {
                "strategy": "sufficient",
                "new_search_terms": [],
                "reasoning": "Unable to generate alternatives after error",
                "confidence": 0.3
            }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Extract JSON if wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Validate required fields
            if "new_search_terms" not in result:
                result["new_search_terms"] = []
            if "reasoning" not in result:
                result["reasoning"] = "Generated via LLM"
            if "confidence" not in result:
                result["confidence"] = 0.7
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {
                "new_search_terms": [],
                "reasoning": "Failed to parse LLM response",
                "confidence": 0.3
            }
    
    def _get_sample_results(self, results: Dict[str, Any], limit: int = 5) -> str:
        """Format sample results for LLM analysis"""
        samples = []
        
        for dataset, data in list(results.get("results", {}).items())[:2]:
            for item in data.get("results", [])[:limit]:
                code = item.get("code", "N/A")
                desc = item.get("description", "No description")
                samples.append(f"- {dataset}: {code} - {desc}")
        
        return "\n".join(samples) if samples else "No results to sample"
    
    def _fallback_broaden(
        self,
        query: str,
        search_history: List[str]
    ) -> Dict[str, Any]:
        """Fallback broadening strategy using simple rules"""
        query_lower = query.lower()
        broader_terms = []
        
        # Remove qualifiers to broaden
        qualifiers = ["acute", "chronic", "primary", "secondary", "severe", "mild"]
        base_query = query
        for qualifier in qualifiers:
            base_query = base_query.replace(qualifier, "").strip()
        
        if base_query != query and base_query not in search_history:
            broader_terms.append(base_query)
        
        # Add common variations
        if "test" in query_lower and "test" not in str(search_history):
            broader_terms.append(query.replace("test", "").strip())
        
        return {
            "strategy": "broaden",
            "new_search_terms": broader_terms[:3],
            "reasoning": "Fallback: removing qualifiers to broaden search",
            "confidence": 0.5
        }
    
    def _fallback_narrow(
        self,
        query: str,
        search_history: List[str]
    ) -> Dict[str, Any]:
        """Fallback narrowing strategy using simple rules"""
        narrower_terms = []
        
        # Add common qualifiers
        qualifiers = ["acute", "chronic", "primary"]
        for qualifier in qualifiers:
            term = f"{qualifier} {query}"
            if term.lower() not in [h.lower() for h in search_history]:
                narrower_terms.append(term)
        
        return {
            "strategy": "narrow",
            "new_search_terms": narrower_terms[:3],
            "reasoning": "Fallback: adding qualifiers to narrow search",
            "confidence": 0.5
        }
