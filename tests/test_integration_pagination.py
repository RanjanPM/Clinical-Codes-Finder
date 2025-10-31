"""
Test Conversation Memory and Pagination Features
Demonstrates the ability to request more results from previous queries
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ClinicalTermLookup
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pagination():
    """Test pagination feature with a query that returns many results"""
    
    print("\n" + "=" * 80)
    print("TESTING CONVERSATION MEMORY & PAGINATION")
    print("=" * 80)
    print()
    
    lookup = ClinicalTermLookup()
    
    # Test 1: Initial query with many results
    print("Test 1: Initial query for 'diabetes' (should return many codes)")
    print("-" * 80)
    results = await lookup.lookup("diabetes")
    print(lookup.format_results(results))
    
    # Check if more results are available
    if results.get("has_more_pages"):
        print("\nSystem detected more results are available")
        
        # Test 2: Request more results
        print("\n" + "=" * 80)
        print("Test 2: Requesting more results with 'show more'")
        print("-" * 80)
        more_results = await lookup.lookup("show more")
        print(lookup.format_results(more_results))
        
        # Test 3: Request even more
        if more_results.get("has_more_pages"):
            print("\n" + "=" * 80)
            print("Test 3: Requesting additional results with 'next'")
            print("-" * 80)
            next_results = await lookup.lookup("next")
            print(lookup.format_results(next_results))
        
        # Test 4: Try to go beyond available results
        print("\n" + "=" * 80)
        print("Test 4: Attempting to get results beyond what's available")
        print("-" * 80)
        # Keep requesting until we hit the end
        for i in range(10):  # Max 10 attempts
            beyond_results = await lookup.lookup("more")
            if beyond_results.get("is_end_of_results"):
                print("Correctly detected end of results")
                print(lookup.format_results(beyond_results))
                break
            elif not beyond_results.get("has_more_pages"):
                print("Last page reached")
                print(lookup.format_results(beyond_results))
                break
    else:
        print("\nWARNING: No additional pages available for this query")
    
    # Test 5: New query should reset memory
    print("\n" + "=" * 80)
    print("Test 5: New query should reset pagination memory")
    print("-" * 80)
    new_results = await lookup.lookup("tuberculosis")
    print(f"New query executed: {new_results.get('query')}")
    print(f"   Page number: {new_results.get('page_number', 'N/A')}")
    print(f"   Is continuation: {new_results.get('is_continuation', False)}")
    
    # Test 6: Memory summary
    print("\n" + "=" * 80)
    print("Test 6: Memory state summary")
    print("-" * 80)
    memory_summary = lookup.memory.get_summary()
    print(f"Memory state: {memory_summary}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


async def test_continuation_keywords():
    """Test different continuation keywords"""
    
    print("\n" + "=" * 80)
    print("TESTING CONTINUATION KEYWORD DETECTION")
    print("=" * 80)
    print()
    
    lookup = ClinicalTermLookup()
    
    # Initial query
    print("Setting up with initial query: 'hypertension'")
    await lookup.lookup("hypertension")
    
    # Test various continuation phrases
    test_phrases = [
        "more",
        "next",
        "show more",
        "continue",
        "more results",
        "show all",
        "additional",
        "rest",
        "see more",
    ]
    
    print("\nTesting continuation keyword detection:")
    print("-" * 80)
    
    for phrase in test_phrases:
        is_continuation = lookup.memory.is_continuation_request(phrase)
        status = "DETECTED" if is_continuation else "NOT DETECTED"
        print(f"{status}: '{phrase}'")
    
    # Test phrases that should NOT be detected as continuation
    print("\nTesting phrases that should NOT trigger continuation:")
    print("-" * 80)
    
    non_continuation_phrases = [
        "diabetes",
        "blood pressure medication",
        "chest pain",
    ]
    
    for phrase in non_continuation_phrases:
        is_continuation = lookup.memory.is_continuation_request(phrase)
        status = "INCORRECTLY DETECTED" if is_continuation else "CORRECTLY IGNORED"
        print(f"{status}: '{phrase}'")
    
    print("\n" + "=" * 80)
    print("KEYWORD DETECTION TESTS COMPLETED")
    print("=" * 80)


async def main():
    """Run all pagination tests"""
    
    print("\n" + "#" * 80)
    print("CONVERSATION MEMORY & PAGINATION TEST SUITE")
    print(f"Configuration: {config.display.MAX_CODES_PER_SYSTEM} codes per page")
    print("#" * 80)
    
    # Run tests
    await test_pagination()
    await test_continuation_keywords()
    
    print("\n" + "#" * 80)
    print("ALL TEST SUITES COMPLETED")
    print("#" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())

