"""
Synthesis Agent
Generates intelligent summaries and clinical insights from search results
"""

import logging
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import config

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Agent for synthesizing insights from medical code results"""
    
    SYSTEM_PROMPT = """You are a medical informatics expert who synthesizes clinical coding results into actionable insights.

Your job is to analyze medical code search results and provide:
1. Executive summary (2-3 sentences)
2. Key patterns and findings
3. Most appropriate codes for common scenarios
4. Clinical context, warnings, or considerations
5. Recommended next steps

CRITICAL FORMATTING RULES:
1. Use SINGLE brackets [DATASETNAME] for all code references - NEVER use double brackets [[]]
2. Use the EXACT dataset name as shown in the input (case-sensitive)
3. For ICD-10-CM codes, prefer ".9" unspecified codes as primary recommendations (e.g., J44.9 for COPD, E11.9 for diabetes)
4. Prioritize official coding systems (ICD-10-CM, LOINC, RxTerms) over general databases (CONDITIONS)

Dataset Reference Guide:
- [ICD10CM] = Official ICD-10-CM diagnosis codes (e.g., J44.9, E11.9, R27.0)
- [ICD11] = ICD-11 codes
- [ICD9CM_DX] = ICD-9-CM diagnosis codes  
- [LOINC] = Laboratory test codes (e.g., 2345-7)
- [RXTERMS] = Medication codes
- [DRUGS] = Prescribable drug ingredients
- [CONDITIONS] = General medical conditions database (supplementary, not for official coding)
- [HPO] = Human Phenotype Ontology for genetic/phenotypic features (e.g., HP:0001251)
- [GENES], [SNPS], [CLINVAR] = Genomic datasets

Example correct formatting:
- [ICD10CM] J44.9
- [LOINC] 2345-7
- [HPO] HP:0001251
X [[ICD10CM]] J44.9  (WRONG - double brackets)
X [ICD-10] J44.9     (WRONG - use ICD10CM)
X [CONDITIONS] 364   (Avoid in top recommendations - use ICD codes instead)

Be concise, accurate, and clinically relevant.
Respond in JSON format with structured insights."""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        self.llm = ChatOpenAI(
            model=model_name or config.llm.SYNTHESIS_MODEL,
            temperature=temperature if temperature is not None else config.llm.SYNTHESIS_TEMPERATURE
        )
    
    async def synthesize_findings(
        self,
        query: str,
        scored_results: Dict[str, Any],
        term_analysis: Dict[str, Any],
        iteration_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive synthesis of search results
        
        Args:
            query: Original user query
            scored_results: Results with relevance scores
            term_analysis: Analysis from terminology agent
            iteration_history: History of search iterations
            
        Returns:
            Dictionary with synthesis insights
        """
        logger.info(f"Synthesizing findings for query: {query}")
        
        # Get quality metrics
        quality_metrics = scored_results.get("quality_metrics", {})
        total_matches = scored_results.get("total_matches", 0)
        high_quality_count = quality_metrics.get("high_quality_count", 0)
        
        # Determine how many recommendations to show: at least all high-quality results
        # but no more than 10
        num_recommendations = max(
            config.display.MAX_TOP_RECOMMENDATIONS,
            min(high_quality_count, 10)
        )
        
        # Get top results across all datasets
        top_results = self._get_top_results(scored_results, limit=num_recommendations)
        
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(
            query=query,
            term_type=term_analysis.get("term_type", "unknown"),
            total_matches=total_matches,
            quality_metrics=quality_metrics,
            top_results=top_results,
            iteration_count=len(iteration_history)
        )
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            synthesis = self._parse_synthesis_response(response.content)
            
            # Add metadata
            synthesis["query"] = query
            synthesis["total_codes_found"] = total_matches
            synthesis["iterations_performed"] = len(iteration_history)
            synthesis["avg_relevance_score"] = quality_metrics.get("avg_relevance", 0.0)
            
            logger.info("Synthesis complete")
            return synthesis
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return self._fallback_synthesis(query, scored_results, term_analysis)
    
    def _build_synthesis_prompt(
        self,
        query: str,
        term_type: str,
        total_matches: int,
        quality_metrics: Dict[str, Any],
        top_results: List[Dict[str, Any]],
        iteration_count: int
    ) -> str:
        """Build prompt for LLM synthesis"""
        
        # Format top results
        results_summary = self._format_results_for_prompt(top_results)
        
        prompt = f"""Analyze these medical coding search results and provide clinical insights.

**User Query:** "{query}"
**Term Type:** {term_type}
**Total Codes Found:** {total_matches}
**Average Relevance:** {quality_metrics.get('avg_relevance', 0):.2f}
**High Quality Results:** {quality_metrics.get('high_quality_count', 0)}
**Iterations Performed:** {iteration_count}

**Top Results:**
{results_summary}

Provide a comprehensive analysis in JSON format with:

{{
    "executive_summary": "2-3 sentence overview of findings",
    "key_patterns": [
        "Notable pattern 1",
        "Notable pattern 2"
    ],
    "top_recommendations": [
        {{
            "code": "just the code value (e.g., 'J44.9' or 'metFORMIN (Oral Pill)') - do NOT include the dataset name",
            "system": "just the dataset name WITHOUT brackets (e.g., 'ICD10CM' or 'RXTERMS')",
            "use_case": "when to use this code",
            "confidence": "high|medium|low"
        }}
    ],
    "clinical_context": "Important clinical considerations, warnings, or context",
    "search_quality": "excellent|good|fair|poor",
    "search_quality_explanation": "why this quality rating",
    "next_steps": [
        "Suggested action 1",
        "Suggested action 2"
    ]
}}

IMPORTANT: Include ALL codes with relevance score >= 0.7 in top_recommendations (there are {quality_metrics.get('high_quality_count', 0)} high-quality results).
Be specific, accurate, and clinically useful."""
        
        return prompt
    
    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Format results for LLM prompt"""
        formatted = []
        
        for i, result in enumerate(results, 1):
            code = result.get("code", "N/A")
            desc = result.get("description", "No description")
            dataset = result.get("dataset", "unknown")
            relevance = result.get("relevance_score", 0)
            
            formatted.append(
                f"{i}. [{dataset.upper()}] {code}: {desc}\n"
                f"   Relevance: {relevance:.2f} ({result.get('relevance_level', 'unknown')})"
            )
        
        return "\n\n".join(formatted) if formatted else "No results to display"
    
    def _parse_synthesis_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON synthesis from LLM response"""
        try:
            # Extract JSON if wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Fix common LLM formatting errors before parsing
            # Remove double brackets that LLM sometimes adds
            content = content.replace("[[", "[").replace("]]", "]")
            
            synthesis = json.loads(content)
            
            # Validate required fields
            required_fields = [
                "executive_summary",
                "key_patterns",
                "clinical_context",
                "search_quality"
            ]
            
            for field in required_fields:
                if field not in synthesis:
                    synthesis[field] = self._get_default_field(field)
            
            # Fix double brackets in recommendations that might have survived JSON parsing
            # LLM sometimes generates [[DATASET]] instead of [DATASET]
            if "top_recommendations" in synthesis:
                for rec in synthesis["top_recommendations"]:
                    if "system" in rec:
                        # Remove ALL brackets first, then re-wrap with single brackets
                        system_name = rec["system"].replace("[", "").replace("]", "").strip()
                        rec["system"] = system_name
            
            return synthesis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse synthesis JSON: {e}")
            logger.debug(f"Raw content: {content}")
            return self._create_simple_synthesis(content)
    
    def _get_default_field(self, field: str) -> Any:
        """Get default value for missing field"""
        defaults = {
            "executive_summary": "Analysis complete. Results retrieved from medical coding databases.",
            "key_patterns": [],
            "top_recommendations": [],
            "clinical_context": "Review results with clinical context in mind.",
            "search_quality": "unknown",
            "search_quality_explanation": "Unable to assess quality",
            "next_steps": ["Review results", "Consult with medical professional if needed"]
        }
        return defaults.get(field, "")
    
    def _create_simple_synthesis(self, content: str) -> Dict[str, Any]:
        """Create simple synthesis from unparsed content"""
        return {
            "executive_summary": content[:200] if content else "Unable to generate synthesis",
            "key_patterns": [],
            "top_recommendations": [],
            "clinical_context": "Manual review recommended",
            "search_quality": "unknown",
            "search_quality_explanation": "Synthesis parsing failed",
            "next_steps": ["Review raw results", "Verify code accuracy"]
        }
    
    def _get_top_results(
        self,
        scored_results: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract top results across all datasets, prioritizing primary coding systems"""
        
        # Define priority order for coding systems (official codes first)
        primary_systems = [
            "icd10cm", "icd11", "icd9cm_dx", "icd9cm_sg",  # Diagnosis/procedure codes (highest priority)
            "loinc",  # Lab tests
            "rxterms", "drugs",  # Medications
            "hcpcs",  # Procedures/services
            "hpo",  # Phenotypes (important for genetics)
            "clinvar", "genes", "snps",  # Genomics
            "pharmvar",  # Pharmacogenomics
            "npi_idv", "npi_org"  # Providers
        ]
        
        # Separate results by priority, with score boost for primary systems
        priority_results = []
        secondary_results = []
        
        for dataset, data in scored_results.get("results", {}).items():
            is_primary = dataset in primary_systems
            
            for item in data.get("results", []):
                # Apply a score boost to primary systems to ensure they rank higher
                # This ensures ICD codes appear in top recommendations even if
                # secondary datasets (like 'conditions') have slightly higher raw scores
                adjusted_item = item.copy()
                if is_primary:
                    # Boost primary system scores by 0.15 for ranking purposes only
                    adjusted_item["ranking_score"] = item.get("relevance_score", 0) + 0.15
                    priority_results.append(adjusted_item)
                else:
                    adjusted_item["ranking_score"] = item.get("relevance_score", 0)
                    secondary_results.append(adjusted_item)
        
        # Sort each group by ranking score (boosted for primary systems)
        priority_results.sort(
            key=lambda x: x.get("ranking_score", 0),
            reverse=True
        )
        secondary_results.sort(
            key=lambda x: x.get("ranking_score", 0),
            reverse=True
        )
        
        # Combine: prioritize primary systems, then add secondary if needed
        all_results = priority_results + secondary_results
        
        return all_results[:limit]
    
    def _fallback_synthesis(
        self,
        query: str,
        scored_results: Dict[str, Any],
        term_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback synthesis when LLM fails"""
        
        total_matches = scored_results.get("total_matches", 0)
        quality_metrics = scored_results.get("quality_metrics", {})
        avg_relevance = quality_metrics.get("avg_relevance", 0)
        
        # Determine search quality
        if total_matches == 0:
            search_quality = "poor"
            quality_explanation = "No results found"
        elif avg_relevance >= 0.7:
            search_quality = "excellent"
            quality_explanation = f"Found {total_matches} highly relevant results"
        elif avg_relevance >= 0.5:
            search_quality = "good"
            quality_explanation = f"Found {total_matches} relevant results"
        else:
            search_quality = "fair"
            quality_explanation = f"Found {total_matches} results with moderate relevance"
        
        # Get top result for recommendation
        top_results = self._get_top_results(scored_results, limit=3)
        recommendations = []
        
        for result in top_results:
            recommendations.append({
                "code": result.get("code", "N/A"),
                "system": result.get("dataset", "unknown"),
                "use_case": "High relevance match for query",
                "confidence": result.get("relevance_level", "medium")
            })
        
        return {
            "executive_summary": (
                f"Search for '{query}' (identified as {term_analysis.get('term_type', 'unknown')}) "
                f"returned {total_matches} results across {len(scored_results.get('results', {}))} "
                f"coding systems with average relevance of {avg_relevance:.2f}."
            ),
            "key_patterns": [
                f"Total matches: {total_matches}",
                f"Coding systems searched: {len(scored_results.get('results', {}))}",
                f"Average relevance: {avg_relevance:.2f}"
            ],
            "top_recommendations": recommendations,
            "clinical_context": (
                "These are automated search results. Always verify code accuracy and "
                "appropriateness for your specific clinical use case."
            ),
            "search_quality": search_quality,
            "search_quality_explanation": quality_explanation,
            "next_steps": [
                "Review top-scored results",
                "Verify codes against official coding guidelines",
                "Consult with coding specialist if needed"
            ]
        }
    
    def format_synthesis_for_display(self, synthesis: Dict[str, Any]) -> str:
        """Format synthesis for console display"""
        output = []
        
        output.append("=" * 80)
        output.append("INTELLIGENT SYNTHESIS")
        output.append("=" * 80)
        output.append("")
        
        # Executive Summary
        output.append("EXECUTIVE SUMMARY")
        output.append("-" * 80)
        exec_summary = synthesis.get("executive_summary", "").strip()
        output.append(exec_summary if exec_summary else "No data available")
        output.append("")
        
        # Search Quality
        quality = synthesis.get("search_quality", "unknown").upper()
        quality_label = {
            "EXCELLENT": "[EXCELLENT]",
            "GOOD": "[GOOD]",
            "FAIR": "[FAIR]",
            "POOR": "[POOR]",
            "UNKNOWN": "[UNKNOWN]"
        }.get(quality, "[UNKNOWN]")
        
        quality_explanation = synthesis.get('search_quality_explanation', "").strip()
        output.append(f"SEARCH QUALITY: {quality_label}")
        output.append(f"   {quality_explanation if quality_explanation else 'No data available'}")
        output.append("")
        
        # Key Patterns
        patterns = synthesis.get("key_patterns", [])
        output.append("KEY PATTERNS")
        output.append("-" * 80)
        if patterns:
            for pattern in patterns[:5]:
                output.append(f"  - {pattern}")
        else:
            output.append("  No data available")
        output.append("")
        
        # Top Recommendations
        recommendations = synthesis.get("top_recommendations", [])
        output.append("TOP RECOMMENDATIONS")
        output.append("-" * 80)
        if recommendations:
            for i, rec in enumerate(recommendations[:3], 1):
                code = rec.get("code", "N/A")
                system = rec.get("system", "unknown").upper()
                use_case = rec.get("use_case", "No use case specified")
                confidence = rec.get("confidence", "medium").upper()
                
                conf_label = {
                    "HIGH": "[HIGH CONFIDENCE]",
                    "MEDIUM": "[MEDIUM CONFIDENCE]",
                    "LOW": "[LOW CONFIDENCE]"
                }.get(confidence, "[UNKNOWN CONFIDENCE]")
                
                output.append(f"  {i}. [{system}] {code}")
                output.append(f"     Use Case: {use_case}")
                output.append(f"     Confidence: {conf_label}")
                output.append("")
        else:
            output.append("  No data available")
            output.append("")
        
        # Clinical Context
        output.append("CLINICAL CONTEXT")
        output.append("-" * 80)
        clinical_context = synthesis.get("clinical_context", "").strip()
        output.append(clinical_context if clinical_context else "No data available")
        output.append("")
        
        # Next Steps
        next_steps = synthesis.get("next_steps", [])
        output.append("NEXT STEPS")
        output.append("-" * 80)
        if next_steps:
            for step in next_steps:
                output.append(f"  - {step}")
        else:
            output.append("  No data available")
        output.append("")
        
        output.append("=" * 80)
        
        return "\n".join(output)
