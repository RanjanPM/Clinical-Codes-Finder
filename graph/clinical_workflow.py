"""
Clinical Term Lookup Workflow
LangGraph workflow for intelligent medical term code lookup with iterative refinement
"""

import logging
from typing import Dict, List, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.terminology_agent import TerminologyAgent
from agents.retrieval_agent import RetrievalAgent
from agents.refinement_agent import SearchRefinementAgent
from agents.scoring_agent import ResultScoringAgent
from agents.synthesis_agent import SynthesisAgent
from config import config

logger = logging.getLogger(__name__)


class ClinicalQueryState(TypedDict):
    """State for the clinical term lookup workflow"""
    query: str
    term_analysis: Optional[Dict[str, Any]]
    retrieval_results: Optional[Dict[str, Any]]
    scored_results: Optional[Dict[str, Any]]
    synthesis: Optional[Dict[str, Any]]
    final_response: Optional[Dict[str, Any]]
    error: Optional[str]
    # Agentic iteration fields
    iteration_count: int
    search_history: List[str]
    result_quality: float
    refinement_strategy: Optional[str]
    iteration_history: List[Dict[str, Any]]


class ClinicalWorkflow:
    """Main workflow for clinical term lookup with agentic iteration"""
    
    # Configuration loaded from config.py (backed by environment variables)
    MAX_ITERATIONS = config.agentic.MAX_ITERATIONS
    MIN_QUALITY_THRESHOLD = config.agentic.MIN_QUALITY_THRESHOLD
    MIN_RESULTS_THRESHOLD = config.agentic.MIN_RESULTS_THRESHOLD
    ENABLE_EARLY_STOPPING = config.agentic.ENABLE_EARLY_STOPPING
    EXCELLENT_QUALITY_THRESHOLD = config.agentic.EXCELLENT_QUALITY_THRESHOLD
    
    def __init__(
        self,
        terminology_agent: TerminologyAgent,
        retrieval_agent: RetrievalAgent,
        refinement_agent: Optional[SearchRefinementAgent] = None,
        scoring_agent: Optional[ResultScoringAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None
    ):
        self.terminology_agent = terminology_agent
        self.retrieval_agent = retrieval_agent
        self.refinement_agent = refinement_agent or SearchRefinementAgent()
        self.scoring_agent = scoring_agent or ResultScoringAgent()
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the iterative LangGraph workflow"""
        workflow = StateGraph(ClinicalQueryState)
        
        # Add nodes
        workflow.add_node("analyze_term", self._analyze_term)
        workflow.add_node("retrieve_codes", self._retrieve_codes)
        workflow.add_node("score_results", self._score_results)
        workflow.add_node("evaluate_quality", self._evaluate_quality)
        workflow.add_node("refine_search", self._refine_search)
        workflow.add_node("synthesize_response", self._synthesize_response)
        
        # Build iterative flow
        workflow.set_entry_point("analyze_term")
        workflow.add_edge("analyze_term", "retrieve_codes")
        workflow.add_edge("retrieve_codes", "score_results")
        workflow.add_edge("score_results", "evaluate_quality")
        
        # Conditional routing based on quality evaluation
        workflow.add_conditional_edges(
            "evaluate_quality",
            self._decide_next_action,
            {
                "refine": "refine_search",      # Need to improve results
                "sufficient": "synthesize_response",  # Results are good enough
                "complete": "synthesize_response"     # Max iterations reached
            }
        )
        
        # Refinement loop back to retrieval
        workflow.add_edge("refine_search", "retrieve_codes")
        workflow.add_edge("synthesize_response", END)
        
        return workflow.compile()
    
    async def _analyze_term(self, state: ClinicalQueryState) -> ClinicalQueryState:
        """Analyze the medical term"""
        logger.info(f"Analyzing term: {state['query']}")
        
        try:
            analysis = self.terminology_agent.analyze_term(state["query"])
            state["term_analysis"] = analysis
            logger.info(f"Term identified as: {analysis.get('term_type')}")
        except Exception as e:
            logger.error(f"Error in term analysis: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _retrieve_codes(self, state: ClinicalQueryState) -> ClinicalQueryState:
        """Retrieve medical codes from appropriate datasets"""
        iteration = state.get("iteration_count", 0)
        logger.info(f"Retrieving codes (iteration {iteration + 1})")
        
        if state.get("error"):
            return state
        
        try:
            analysis = state.get("term_analysis", {})
            primary_datasets = analysis.get("primary_datasets", ["icd10cm"])
            secondary_datasets = analysis.get("secondary_datasets", [])
            
            # Use refined search terms if available, otherwise use original analysis
            refinement = state.get("refinement_strategy")
            if refinement and iteration > 0:
                search_terms = analysis.get("refined_search_terms", analysis.get("search_terms", [state["query"]]))
                logger.info(f"Using refined search terms: {search_terms[:3]}")
            else:
                search_terms = analysis.get("search_terms", [state["query"]])
            
            # Normalize dataset names to lowercase
            primary_datasets = [d.lower() for d in primary_datasets]
            secondary_datasets = [d.lower() for d in secondary_datasets]
            
            # Sequential retrieval with early stopping
            all_datasets = list(set(primary_datasets + secondary_datasets))
            
            # Use the first generated search term (most formal) as primary, then alternatives
            # This ensures we search with the best term (e.g., "chronic obstructive pulmonary disease")
            # instead of the user's abbreviation (e.g., "copd")
            primary_search_term = search_terms[0] if search_terms else state["query"]
            alternative_search_terms = search_terms[1:] if len(search_terms) > 1 else []
            
            # Also include original query if it's different from generated terms
            if state["query"].lower() not in [t.lower() for t in search_terms]:
                alternative_search_terms.append(state["query"])
            
            results = await self.retrieval_agent.retrieve_with_alternatives(
                term=primary_search_term,
                alternative_terms=alternative_search_terms,
                datasets=all_datasets,
                max_results=config.display.MAX_CODES_PER_SYSTEM
            )
            
            state["retrieval_results"] = results
            
            # Track search history
            if "search_history" not in state:
                state["search_history"] = []
            state["search_history"].extend(search_terms)
            
            logger.info(f"Retrieved {results.get('total_matches', 0)} matches")
            
        except Exception as e:
            logger.error(f"Error in code retrieval: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _score_results(self, state: ClinicalQueryState) -> ClinicalQueryState:
        """Score results for relevance"""
        logger.info("Scoring results for relevance")
        
        if state.get("error") or not state.get("retrieval_results"):
            return state
        
        try:
            scored_results = await self.scoring_agent.score_results(
                query=state["query"],
                results=state["retrieval_results"],
                term_analysis=state.get("term_analysis", {})
            )
            
            state["scored_results"] = scored_results
            
            quality_metrics = scored_results.get("quality_metrics", {})
            logger.info(f"Scoring complete. Avg relevance: {quality_metrics.get('avg_relevance', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error in result scoring: {e}")
            # Fall back to unscored results
            state["scored_results"] = state["retrieval_results"]
        
        return state
    
    async def _evaluate_quality(self, state: ClinicalQueryState) -> ClinicalQueryState:
        """Evaluate result quality and decide if refinement is needed"""
        iteration = state.get("iteration_count", 0)
        scored_results = state.get("scored_results", {})
        
        # Calculate quality metrics
        total_matches = scored_results.get("total_matches", 0)
        quality_metrics = scored_results.get("quality_metrics", {})
        avg_relevance = quality_metrics.get("avg_relevance", 0)
        high_quality_count = quality_metrics.get("high_quality_count", 0)
        
        # Calculate overall quality score
        if total_matches == 0:
            quality_score = 0.0
        else:
            # Weighted quality using configured weights
            relevance_component = avg_relevance * config.agentic.QUALITY_RELEVANCE_WEIGHT
            count_component = min(1.0, total_matches / 10.0) * config.agentic.QUALITY_COUNT_WEIGHT
            quality_score = relevance_component + count_component
        
        state["result_quality"] = quality_score
        state["iteration_count"] = iteration + 1
        
        # Record iteration history
        if "iteration_history" not in state:
            state["iteration_history"] = []
        
        state["iteration_history"].append({
            "iteration": iteration + 1,
            "total_matches": total_matches,
            "avg_relevance": avg_relevance,
            "quality_score": quality_score,
            "high_quality_count": high_quality_count
        })
        
        logger.info(
            f"Quality evaluation (iteration {iteration + 1}): "
            f"score={quality_score:.2f}, matches={total_matches}, "
            f"avg_relevance={avg_relevance:.2f}"
        )
        
        # Decision logic
        if iteration >= self.MAX_ITERATIONS:
            state["refinement_strategy"] = "complete"
            logger.info("Max iterations reached, proceeding to synthesis")
        elif (self.ENABLE_EARLY_STOPPING and 
              quality_score >= self.EXCELLENT_QUALITY_THRESHOLD and 
              total_matches >= self.MIN_RESULTS_THRESHOLD):
            state["refinement_strategy"] = "sufficient"
            logger.info(f"Excellent quality ({quality_score:.2f}) achieved, early stopping")
        elif quality_score >= self.MIN_QUALITY_THRESHOLD and total_matches >= self.MIN_RESULTS_THRESHOLD:
            state["refinement_strategy"] = "sufficient"
            logger.info("Quality threshold met, proceeding to synthesis")
        elif total_matches == 0 or quality_score < 0.3:
            state["refinement_strategy"] = "refine"
            logger.info("Quality insufficient, will refine search")
        else:
            state["refinement_strategy"] = "sufficient"
            logger.info("Acceptable quality, proceeding to synthesis")
        
        return state
    
    async def _refine_search(self, state: ClinicalQueryState) -> ClinicalQueryState:
        """Refine search strategy based on previous results"""
        logger.info("Refining search strategy")
        
        if state.get("error"):
            return state
        
        try:
            refinement = await self.refinement_agent.refine_strategy(
                original_query=state["query"],
                term_type=state.get("term_analysis", {}).get("term_type", "unknown"),
                previous_results=state.get("scored_results", {}),
                iteration=state.get("iteration_count", 0),
                search_history=state.get("search_history", [])
            )
            
            # Update term analysis with refined search terms
            term_analysis = state.get("term_analysis", {})
            term_analysis["refined_search_terms"] = refinement.get("new_search_terms", [])
            term_analysis["refinement_reasoning"] = refinement.get("reasoning", "")
            state["term_analysis"] = term_analysis
            
            logger.info(
                f"Refinement strategy: {refinement.get('strategy', 'unknown')}, "
                f"new terms: {refinement.get('new_search_terms', [])[:3]}"
            )
            
        except Exception as e:
            logger.error(f"Error in search refinement: {e}")
            # Mark as complete to avoid infinite loop
            state["refinement_strategy"] = "complete"
        
        return state
    
    def _decide_next_action(self, state: ClinicalQueryState) -> str:
        """Conditional edge router based on refinement strategy"""
        strategy = state.get("refinement_strategy", "sufficient")
        
        if strategy == "refine":
            return "refine"
        elif strategy == "complete":
            return "complete"
        else:
            return "sufficient"
    
    async def _synthesize_response(self, state: ClinicalQueryState) -> ClinicalQueryState:
        """Synthesize final response with intelligent insights"""
        logger.info("Synthesizing final response with insights")
        
        if state.get("error"):
            state["final_response"] = {
                "success": False,
                "error": state["error"],
                "query": state["query"]
            }
            return state
        
        try:
            analysis = state.get("term_analysis", {})
            scored_results = state.get("scored_results", state.get("retrieval_results", {}))
            iteration_history = state.get("iteration_history", [])
            
            # Generate intelligent synthesis
            synthesis = await self.synthesis_agent.synthesize_findings(
                query=state["query"],
                scored_results=scored_results,
                term_analysis=analysis,
                iteration_history=iteration_history
            )
            
            state["synthesis"] = synthesis
            
            # Structure the final response
            final_response = {
                "success": True,
                "query": state["query"],
                "term_type": analysis.get("term_type"),
                "confidence": analysis.get("confidence"),
                "reasoning": analysis.get("reasoning"),
                "primary_datasets": analysis.get("primary_datasets", []),
                "total_matches": scored_results.get("total_matches", 0),
                "datasets_searched": scored_results.get("datasets_searched", 0),
                "codes_by_system": self._organize_by_coding_system(scored_results),
                "quality_metrics": scored_results.get("quality_metrics", {}),
                "synthesis": synthesis,
                "iteration_count": state.get("iteration_count", 1),
                "iteration_history": iteration_history,
                "result_quality": state.get("result_quality", 0)
            }
            
            state["final_response"] = final_response
            
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            # Fallback to basic response
            results = state.get("scored_results", state.get("retrieval_results", {}))
            state["final_response"] = {
                "success": True,
                "query": state["query"],
                "term_type": analysis.get("term_type"),
                "confidence": analysis.get("confidence"),
                "reasoning": analysis.get("reasoning"),
                "total_matches": results.get("total_matches", 0),
                "datasets_searched": results.get("datasets_searched", 0),
                "codes_by_system": self._organize_by_coding_system(results),
                "synthesis": {"executive_summary": "Unable to generate detailed synthesis"},
                "error_details": str(e)
            }
        
        return state
    
    def _organize_by_coding_system(self, results: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Organize results by coding system for better presentation"""
        organized = {}
        
        coding_system_names = {
            "icd10cm": "ICD-10-CM",
            "icd11": "ICD-11",
            "icd9cm_dx": "ICD-9-CM Diagnoses",
            "icd9cm_sg": "ICD-9-CM Procedures",
            "loinc": "LOINC",
            "rxterms": "RxTerms",
            "drugs": "Drugs",
            "hcpcs": "HCPCS",
            "ucum": "UCUM",
            "hpo": "HPO",
            "conditions": "Medical Conditions",
            "procedures": "Procedures",
            "clinvar": "ClinVar",
            "genes": "Genes",
            "snps": "SNPs",
            "genetic_diseases": "Genetic Diseases",
            "pharmvar": "PharmVar",
            "npi_idv": "NPI (Individuals)",
            "npi_org": "NPI (Organizations)"
        }
        
        for dataset, data in results.get("results", {}).items():
            system_name = coding_system_names.get(dataset, dataset.upper())
            
            # Append results to system name, handling duplicates
            if system_name not in organized:
                organized[system_name] = []
            
            organized[system_name].extend([
                {
                    "code": item.get("code", "N/A"),
                    "description": item.get("description", "No description available"),
                    "dataset": dataset,
                    "relevance_score": item.get("relevance_score"),
                    "relevance_level": item.get("relevance_level")
                }
                for item in data.get("results", [])
            ])
        
        return organized
    
    async def run(self, query: str) -> Dict[str, Any]:
        """
        Run the iterative agentic workflow for a clinical term lookup
        
        Args:
            query: The clinical term to look up
            
        Returns:
            Dictionary with lookup results including synthesis and iteration history
        """
        initial_state: ClinicalQueryState = {
            "query": query,
            "term_analysis": None,
            "retrieval_results": None,
            "scored_results": None,
            "synthesis": None,
            "final_response": None,
            "error": None,
            "iteration_count": 0,
            "search_history": [],
            "result_quality": 0.0,
            "refinement_strategy": None,
            "iteration_history": []
        }
        
        logger.info(f"Starting agentic workflow for query: {query}")
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            result = final_state.get("final_response", {})
            
            logger.info(
                f"Workflow complete: {result.get('iteration_count', 0)} iterations, "
                f"quality={result.get('result_quality', 0):.2f}"
            )
            
            return result
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
