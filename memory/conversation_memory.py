"""
Conversation Memory Module
Maintains conversation history and supports pagination of previous results
"""

import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history and result pagination"""
    
    # Keywords that indicate user wants to see more results
    CONTINUATION_KEYWORDS = [
        "more", "next", "continue", "show more", "additional", 
        "rest", "others", "see more", "keep going", "more results",
        "show all", "full list", "complete list", "everything"
    ]
    
    # Cache TTL for full query results (in seconds)
    QUERY_CACHE_TTL = 3600  # 1 hour
    
    def __init__(self):
        """Initialize conversation memory"""
        self.last_query: Optional[str] = None
        self.last_results: Optional[Dict[str, Any]] = None
        self.current_page: int = 0
        self.codes_per_page: int = 5  # Default, will be updated from config
        self.query_timestamp: Optional[datetime] = None
        
        # Query result cache: {query_hash: (results, timestamp)}
        self.query_cache: Dict[str, tuple[Dict[str, Any], datetime]] = {}
        
        logger.info("Conversation memory initialized with query result caching")
    
    def _generate_query_hash(self, query: str) -> str:
        """
        Generate a hash for query caching
        
        Args:
            query: User query string
            
        Returns:
            MD5 hash of normalized query
        """
        # Normalize query: lowercase, strip whitespace
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_cached_query(self, query: str) -> bool:
        """
        Check if this exact query has cached results
        
        Args:
            query: User query string
            
        Returns:
            True if valid cached results exist
        """
        query_hash = self._generate_query_hash(query)
        
        if query_hash not in self.query_cache:
            return False
        
        # Check if cache is still valid (within TTL)
        _, timestamp = self.query_cache[query_hash]
        age = datetime.now() - timestamp
        
        if age.total_seconds() > self.QUERY_CACHE_TTL:
            # Cache expired, remove it
            del self.query_cache[query_hash]
            logger.info(f"Query cache expired for: {query[:50]}")
            return False
        
        return True
    
    def get_cached_results(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results for a query
        
        Args:
            query: User query string
            
        Returns:
            Cached results dictionary or None if not found/expired
        """
        if not self.is_cached_query(query):
            return None
        
        query_hash = self._generate_query_hash(query)
        results, timestamp = self.query_cache[query_hash]
        
        age_seconds = (datetime.now() - timestamp).total_seconds()
        logger.info(
            f"Retrieved cached results for query '{query[:50]}' "
            f"(age: {age_seconds:.0f}s)"
        )
        
        # Add cache metadata
        cached_results = results.copy()
        cached_results["from_cache"] = True
        cached_results["cache_age_seconds"] = age_seconds
        
        return cached_results
    
    def cache_query_results(self, query: str, results: Dict[str, Any]):
        """
        Cache full query results
        
        Args:
            query: User query string
            results: Complete results dictionary
        """
        query_hash = self._generate_query_hash(query)
        self.query_cache[query_hash] = (results.copy(), datetime.now())
        
        logger.info(
            f"Cached query results for: {query[:50]} "
            f"(cache size: {len(self.query_cache)})"
        )
        
        # Cleanup old cache entries (keep max 100 queries)
        if len(self.query_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self.query_cache.items(), 
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            # Keep only the 50 most recent
            self.query_cache = dict(sorted_items[-50:])
            logger.info(f"Cleaned up query cache, kept 50 most recent entries")
    
    def is_continuation_request(self, user_input: str) -> bool:
        """
        Check if user input is requesting more results from previous query
        
        Args:
            user_input: User's input text
            
        Returns:
            True if this is a continuation request
        """
        if not self.last_query or not self.last_results:
            return False
        
        # Convert to lowercase for comparison
        input_lower = user_input.lower().strip()
        
        # Check for direct continuation keywords
        if any(keyword in input_lower for keyword in self.CONTINUATION_KEYWORDS):
            # Make sure it's not a new clinical term that happens to contain "more"
            # (e.g., "Baltimore criteria" contains "more")
            word_count = len(input_lower.split())
            if word_count <= 3:  # Short phrases are likely continuation requests
                return True
        
        return False
    
    def store_results(self, query: str, results: Dict[str, Any], codes_per_page: int):
        """
        Store results from a new query
        
        Args:
            query: The original user query
            results: Full results dictionary
            codes_per_page: Number of codes to show per page
        """
        self.last_query = query
        self.last_results = results
        self.current_page = 0
        self.codes_per_page = codes_per_page
        self.query_timestamp = datetime.now()
        
        # Count total available codes
        total_codes = 0
        codes_by_system = results.get("codes_by_system", {})
        for codes in codes_by_system.values():
            total_codes += len(codes)
        
        logger.info(
            f"Stored results for query '{query}': "
            f"{total_codes} total codes across {len(codes_by_system)} systems"
        )
    
    def get_next_page(self) -> Optional[Dict[str, Any]]:
        """
        Get the next page of results from the last query
        
        Returns:
            Modified results dictionary with next page of codes, or None if no more pages
        """
        if not self.last_results:
            return None
        
        self.current_page += 1
        return self._get_page(self.current_page)
    
    def _get_page(self, page_number: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific page of results
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            Modified results dictionary with requested page of codes
        """
        if not self.last_results:
            return None
        
        codes_by_system = self.last_results.get("codes_by_system", {})
        if not codes_by_system:
            return None
        
        # Calculate pagination for each system
        paginated_results = {}
        pagination_info = {}  # Track range for each system
        has_more_results = False
        total_shown = 0
        total_available = 0
        
        for system_name, all_codes in codes_by_system.items():
            total_codes_in_system = len(all_codes)
            total_available += total_codes_in_system
            
            start_idx = page_number * self.codes_per_page
            end_idx = start_idx + self.codes_per_page
            
            # Get codes for this page
            page_codes = all_codes[start_idx:end_idx]
            
            if page_codes:
                paginated_results[system_name] = page_codes
                total_shown += len(page_codes)
                
                # Store range information for this system
                pagination_info[system_name] = {
                    "start": start_idx + 1,  # 1-indexed for display
                    "end": min(end_idx, total_codes_in_system),
                    "total": total_codes_in_system
                }
                
                # Check if there are more codes in this system
                if end_idx < len(all_codes):
                    has_more_results = True
        
        if not paginated_results:
            logger.info(f"No more results for page {page_number}")
            return None
        
        # Create modified results for this page
        page_results = self.last_results.copy()
        page_results["codes_by_system"] = paginated_results
        page_results["pagination_info"] = pagination_info
        page_results["page_number"] = page_number + 1
        page_results["is_continuation"] = True
        page_results["has_more_pages"] = has_more_results
        page_results["total_codes_shown"] = total_shown
        page_results["total_codes_available"] = total_available
        
        logger.info(
            f"Retrieved page {page_number + 1}: "
            f"showing {total_shown} codes ({total_available} total available)"
        )
        
        return page_results
    
    def reset(self):
        """Clear conversation memory (but keep query cache)"""
        self.last_query = None
        self.last_results = None
        self.current_page = 0
        self.query_timestamp = None
        logger.info("Conversation memory reset (query cache preserved)")
    
    def clear_cache(self):
        """Clear the query result cache"""
        cache_size = len(self.query_cache)
        self.query_cache.clear()
        logger.info(f"Cleared query cache ({cache_size} entries removed)")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state"""
        if not self.last_query:
            return {
                "has_memory": False,
                "cache_size": len(self.query_cache),
                "cache_enabled": True
            }
        
        codes_by_system = self.last_results.get("codes_by_system", {}) if self.last_results else {}
        total_codes = sum(len(codes) for codes in codes_by_system.values())
        
        return {
            "has_memory": True,
            "last_query": self.last_query,
            "current_page": self.current_page + 1,
            "codes_per_page": self.codes_per_page,
            "total_systems": len(codes_by_system),
            "total_codes": total_codes,
            "timestamp": self.query_timestamp.isoformat() if self.query_timestamp else None,
            "cache_size": len(self.query_cache),
            "cache_enabled": True
        }
