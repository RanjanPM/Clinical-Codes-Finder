"""
Clinical Term Lookup System - Main Application
Interactive system for looking up medical codes across coding systems
"""

import asyncio
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from agents.terminology_agent import TerminologyAgent
from agents.retrieval_agent import RetrievalAgent
from agents.refinement_agent import SearchRefinementAgent
from agents.scoring_agent import ResultScoringAgent
from agents.synthesis_agent import SynthesisAgent
from apis.clinical_tables import ClinicalTablesClient
from graph.clinical_workflow import ClinicalWorkflow
from memory.conversation_memory import ConversationMemory
from config import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.display.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalTermLookup:
    """Main application class for clinical term lookup"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the clinical term lookup system"""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Initialize components
        self.client = ClinicalTablesClient()
        self.terminology_agent = TerminologyAgent()
        self.retrieval_agent = RetrievalAgent(self.client)
        self.refinement_agent = SearchRefinementAgent()
        self.scoring_agent = ResultScoringAgent()
        self.synthesis_agent = SynthesisAgent()
        
        self.workflow = ClinicalWorkflow(
            self.terminology_agent,
            self.retrieval_agent,
            self.refinement_agent,
            self.scoring_agent,
            self.synthesis_agent
        )
        
        # Initialize conversation memory for pagination
        self.memory = ConversationMemory()
        
        logger.info("Clinical Term Lookup System initialized")
    
    async def lookup(self, term: str, check_continuation: bool = True) -> dict:
        """
        Look up medical codes for a clinical term
        
        Args:
            term: The clinical term (e.g., "blood sugar test", "tuberculosis")
            check_continuation: Whether to check if this is a continuation request
            
        Returns:
            Dictionary with codes across multiple coding systems
        """
        # Check if this is a request for more results from previous query
        if check_continuation and self.memory.is_continuation_request(term):
            logger.info(f"Detected continuation request: '{term}'")
            next_page = self.memory.get_next_page()
            
            if next_page:
                logger.info(f"Returning page {next_page.get('page_number')} for previous query")
                return next_page
            else:
                # No more results
                return {
                    "success": False,
                    "error": f"No more results available for '{self.memory.last_query}'",
                    "is_end_of_results": True,
                    "last_query": self.memory.last_query
                }
        
        # Check if we have cached results for this exact query
        if self.memory.is_cached_query(term):
            cached_results = self.memory.get_cached_results(term)
            if cached_results:
                logger.info(f"Returning cached results for: {term}")
                
                # Add fresh pagination info for display
                codes_by_system = cached_results.get("codes_by_system", {})
                pagination_info = {}
                for system_name, codes in codes_by_system.items():
                    total = len(codes)
                    shown = min(config.display.MAX_CODES_PER_SYSTEM, total)
                    pagination_info[system_name] = {
                        "start": 1,
                        "end": shown,
                        "total": total
                    }
                cached_results["pagination_info"] = pagination_info
                
                # Store in memory for pagination
                self.memory.store_results(
                    term, 
                    cached_results, 
                    config.display.MAX_CODES_PER_SYSTEM
                )
                
                return cached_results
        
        logger.info(f"Looking up: {term}")
        
        async with self.client:
            result = await self.workflow.run(term)
        
        # Store results in memory for potential pagination
        if result.get("success"):
            # Cache the full query results
            self.memory.cache_query_results(term, result)
            
            self.memory.store_results(
                term, 
                result, 
                config.display.MAX_CODES_PER_SYSTEM
            )
            
            # Check if there are more pages available and add pagination info
            codes_by_system = result.get("codes_by_system", {})
            has_more = any(
                len(codes) > config.display.MAX_CODES_PER_SYSTEM 
                for codes in codes_by_system.values()
            )
            result["has_more_pages"] = has_more
            
            # Add pagination info for first page
            pagination_info = {}
            for system_name, codes in codes_by_system.items():
                total = len(codes)
                shown = min(config.display.MAX_CODES_PER_SYSTEM, total)
                pagination_info[system_name] = {
                    "start": 1,
                    "end": shown,
                    "total": total
                }
            result["pagination_info"] = pagination_info
        
        return result
    
    def format_results(self, results: dict) -> str:
        """Format results for display with synthesis and quality metrics"""
        if not results.get("success"):
            # Check if this is end of pagination
            if results.get("is_end_of_results"):
                return (
                    f"\nNo more results available.\n"
                    f"All codes for '{results.get('last_query')}' have been displayed.\n"
                )
            return f"Error: {results.get('error', 'Unknown error')}\n"
        
        output = []
        
        # Header section - different for continuation pages
        if results.get("is_continuation"):
            output.append("=" * 80)
            output.append(f"Query: {results['query']} (CONTINUED - Page {results.get('page_number', 1)})")
            output.append(f"Showing codes {results.get('total_codes_shown', 0)} more codes...")
            output.append("=" * 80)
            output.append("")
        else:
            output.append("=" * 80)
            output.append(f"Query: {results['query']}")
            output.append(f"Term Type: {results.get('term_type', 'Unknown')}")
            
            # Show cache indicator
            if results.get("from_cache"):
                cache_age = results.get("cache_age_seconds", 0)
                if cache_age < 60:
                    age_str = f"{cache_age:.0f} seconds"
                else:
                    age_str = f"{cache_age/60:.1f} minutes"
                output.append(f"Source: Cached results (age: {age_str})")
                output.append("Note: No API calls or LLM processing - instant retrieval")
            output.append(f"Confidence: {results.get('confidence', 0):.2%}")
            output.append(f"Reasoning: {results.get('reasoning', 'N/A')}")
            output.append("=" * 80)
            output.append("")
        
        # Agentic metrics (skip for continuation pages)
        if not results.get("is_continuation") and config.display.SHOW_QUALITY_METRICS:
            iteration_count = results.get("iteration_count", 1)
            result_quality = results.get("result_quality", 0)
            quality_metrics = results.get("quality_metrics", {})
            primary_datasets = results.get("primary_datasets", [])
            
            output.append("AGENTIC WORKFLOW METRICS")
            output.append("-" * 80)
            
            # Dataset selection strategy
            if primary_datasets:
                dataset_names = [ds.upper().replace("_", "-") for ds in primary_datasets]
                output.append(f"  Dataset Selection: {len(dataset_names)} systems chosen based on term type '{results.get('term_type', 'unknown')}'")
                output.append(f"  Selected Systems: {', '.join(dataset_names)}")
            
            output.append(f"  Iterations Performed: {iteration_count}")
            output.append(f"  Result Quality Score: {result_quality:.2%}")
            output.append(f"  Total Matches: {results.get('total_matches', 0)}")
            output.append(f"  Average Relevance: {quality_metrics.get('avg_relevance', 0):.2%}")
            output.append(f"  High Quality Results: {quality_metrics.get('high_quality_count', 0)}")
            output.append("")
        
        # Iteration history (skip for continuation pages)
        if not results.get("is_continuation") and config.display.SHOW_ITERATION_HISTORY:
            iteration_history = results.get("iteration_history", [])
            if len(iteration_history) > 1:
                output.append("ITERATION HISTORY")
                output.append("-" * 80)
                for hist in iteration_history:
                    output.append(
                        f"  Iteration {hist['iteration']}: "
                        f"{hist['total_matches']} matches, "
                        f"quality={hist['quality_score']:.2f}, "
                        f"avg_relevance={hist['avg_relevance']:.2f}"
                    )
                output.append("")
        
        # Synthesis (skip for continuation pages)
        if not results.get("is_continuation") and config.display.SHOW_SYNTHESIS:
            synthesis = results.get("synthesis", {})
            if synthesis:
                output.append(self.synthesis_agent.format_synthesis_for_display(synthesis))
                output.append("")
        
        # Detailed codes by system
        codes_by_system = results.get("codes_by_system", {})
        
        if not codes_by_system:
            output.append("No codes found in any coding system.")
            return "\n".join(output)
        
        output.append(f"DETAILED CODES ({len(codes_by_system)} coding system(s))")
        output.append("=" * 80)
        output.append("")
        
        # Get pagination info
        pagination_info = results.get("pagination_info", {})
        
        for system_name, codes in codes_by_system.items():
            # Get range information for this system
            page_info = pagination_info.get(system_name, {})
            if page_info:
                start = page_info.get("start", 1)
                end = page_info.get("end", len(codes))
                total = page_info.get("total", len(codes))
                output.append(f"{system_name} (showing {start}-{end} of {total} results)")
            else:
                output.append(f"{system_name} ({len(codes)} results)")
            output.append("-" * 80)
            
            for i, code_info in enumerate(codes[:config.display.MAX_CODES_PER_SYSTEM], 1):
                code = code_info.get("code", "N/A")
                desc = code_info.get("description", "No description")
                relevance = code_info.get("relevance_score")
                relevance_level = code_info.get("relevance_level", "unknown")
                
                # Relevance indicator
                if config.display.SHOW_RELEVANCE_SCORES and relevance is not None:
                    relevance_label = {
                        "high": "[HIGH]",
                        "medium": "[MEDIUM]",
                        "low": "[LOW]",
                        "very_low": "[VERY LOW]"
                    }.get(relevance_level, "")
                    output.append(f"  {i}. {relevance_label} Code: {code} (Relevance: {relevance:.2f})")
                else:
                    output.append(f"  {i}. Code: {code}")
                
                output.append(f"     Description: {desc}")
                output.append("")
        
        # Pagination hint
        if results.get("has_more_pages"):
            output.append("")
            output.append("TIP: More results available!")
            output.append("     Type 'more', 'next', or 'show more' to see additional codes.")
            output.append("")
        
        # Add disclaimer (only for first page)
        if not results.get("is_continuation"):
            output.append("")
            output.append("=" * 80)
            output.append("IMPORTANT DISCLAIMER")
            output.append("=" * 80)
        output.append("This AI-powered tool is for informational and research purposes only.")
        output.append("")
        output.append("LIMITATIONS:")
        output.append("  - AI-generated analysis may contain errors or inaccuracies")
        output.append("  - Medical codes and recommendations require verification")
        output.append("  - Not a substitute for professional medical judgment")
        output.append("  - Clinical Tables API data quality may vary by dataset")
        output.append("")
        output.append("REQUIRED ACTIONS:")
        output.append("  - All results MUST be reviewed by qualified healthcare professionals")
        output.append("  - Verify all medical codes against official coding guidelines")
        output.append("  - Consult current ICD, LOINC, and other official coding manuals")
        output.append("  - Do not use for direct patient care without human oversight")
        output.append("")
        output.append("COMPLIANCE:")
        output.append("  - Users are responsible for HIPAA compliance and data privacy")
        output.append("  - No PHI (Protected Health Information) should be entered into this system")
        output.append("  - Medical liability rests with the healthcare provider, not this tool")
        output.append("=" * 80)
        
        return "\n".join(output)
    
    async def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "=" * 80)
        print("Clinical Term Lookup System")
        print("=" * 80)
        print("Enter clinical terms to find relevant medical codes across coding systems.")
        print("Examples: 'blood sugar test', 'tuberculosis', 'metformin 500 mg'")
        print("\nTIP: If more results are available, type 'more' or 'next' to see them.")
        print("Special commands: 'clear cache' to clear query cache")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                term = input("Enter clinical term: ").strip()
                
                if term.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Check for special commands
                if term.lower() in ['clear cache', 'clear', 'reset cache']:
                    cache_size = len(self.memory.query_cache)
                    self.memory.clear_cache()
                    print(f"\nCleared query cache ({cache_size} entries removed)")
                    print("All future queries will be processed fresh.\n")
                    continue
                
                if term.lower() == 'cache status':
                    summary = self.memory.get_summary()
                    print(f"\nCache Status:")
                    print(f"  Cached queries: {summary['cache_size']}")
                    print(f"  Cache enabled: {summary['cache_enabled']}")
                    if summary.get('has_memory'):
                        print(f"  Last query: {summary['last_query']}")
                        print(f"  Current page: {summary['current_page']}")
                    print()
                    continue
                
                if not term:
                    print("Please enter a term.\n")
                    continue
                
                # Show searching message only if not from cache
                if not self.memory.is_cached_query(term):
                    print(f"\nSearching medical codes for '{term}'...\n")
                else:
                    print(f"\nRetrieving cached results for '{term}'...\n")
                
                results = await self.lookup(term)
                print(self.format_results(results))
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {e}\n")


async def main():
    """Main entry point"""
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in environment variables")
            print("Please set your OpenAI API key:")
            print("  1. Create a .env file in the project root")
            print("  2. Add: OPENAI_API_KEY=your_key_here")
            return
        
        lookup_system = ClinicalTermLookup()
        
        # Check if running with command-line argument
        import sys
        if len(sys.argv) > 1:
            # Single lookup mode
            term = " ".join(sys.argv[1:])
            results = await lookup_system.lookup(term)
            print(lookup_system.format_results(results))
        else:
            # Interactive mode
            await lookup_system.interactive_mode()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
