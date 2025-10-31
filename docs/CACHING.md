# Query Result Caching

## Overview

The Clinical Term Lookup System implements **full query result caching** to provide instant responses for repeated queries without re-processing or making API calls.

## How It Works

### Multi-Level Caching Architecture

The system uses a **two-tier caching strategy**:

1. **API-Level Caching** (in `apis/clinical_tables.py`)
   - Caches raw API responses from Clinical Tables
   - TTL: Configurable (default: 24 hours for stable medical codes)
   - Scope: Individual API endpoint calls
   - Purpose: Reduce network calls to Clinical Tables API

2. **Query Result Caching** (in `memory/conversation_memory.py`)
   - Caches complete query results after full agent processing
   - TTL: 1 hour (3600 seconds)
   - Scope: End-to-end query results
   - Purpose: Skip ALL processing for identical queries

### Performance Benefits

When a cached query is submitted:
- ✅ **No LLM API calls** (saves OpenAI costs)
- ✅ **No agent processing** (terminology, retrieval, scoring, synthesis)
- ✅ **No Clinical Tables API calls** (even if API cache expired)
- ✅ **Instant response** (typically <0.01 seconds vs 15-30 seconds)

**Measured Speedup**: ~16,000x faster for cached queries!

## Cache Behavior

### Query Matching

Queries are normalized before caching:
```python
# These are considered IDENTICAL:
"diabetes"
"Diabetes"
"  diabetes  "
" DIABETES "
```

Query hash is generated using MD5 of normalized (lowercase, stripped) query text.

### Cache Expiration

- **TTL**: 1 hour per query
- **Auto-cleanup**: Keeps max 100 queries in cache (removes oldest when exceeded)
- **Manual clearing**: Use `clear cache` command in interactive mode

### What Gets Cached

The complete result dictionary including:
- All coding system results (ICD-10-CM, LOINC, conditions, etc.)
- Relevance scores
- Synthesis findings
- Term analysis
- Quality metrics
- All metadata

### What Is NOT Cached

- Pagination state (regenerated fresh)
- "More results" continuation requests (uses separate pagination logic)

## Usage

### Automatic Caching

Caching happens automatically - no configuration required:

```python
lookup = ClinicalTermLookup()

# First query - full processing (~20 seconds)
result1 = await lookup.lookup("diabetes")

# Second identical query - instant from cache (<0.01 seconds)
result2 = await lookup.lookup("diabetes")
```

### Interactive Mode Commands

```bash
# Clear the cache
Enter clinical term: clear cache

# Check cache status
Enter clinical term: cache status

# Results show cache indicator
Query: diabetes
Term Type: diagnosis
Source: Cached results (age: 5.2 seconds)
Note: No API calls or LLM processing - instant retrieval
```

### Programmatic Cache Control

```python
# Check if query has cached results
if lookup.memory.is_cached_query("diabetes"):
    print("Results available from cache")

# Get cached results directly
cached = lookup.memory.get_cached_results("diabetes")

# Clear cache manually
lookup.memory.clear_cache()

# Get cache statistics
summary = lookup.memory.get_summary()
print(f"Cached queries: {summary['cache_size']}")
```

## Configuration

### Adjust Cache TTL

Edit `memory/conversation_memory.py`:

```python
class ConversationMemory:
    # Cache TTL in seconds
    QUERY_CACHE_TTL = 3600  # 1 hour (default)
    # QUERY_CACHE_TTL = 7200  # 2 hours
    # QUERY_CACHE_TTL = 86400  # 24 hours
```

### Adjust Max Cache Size

```python
def cache_query_results(self, query: str, results: Dict[str, Any]):
    # Cleanup threshold
    if len(self.query_cache) > 100:  # Change this number
        # Keep only the 50 most recent
        self.query_cache = dict(sorted_items[-50:])  # Adjust retention
```

## Testing

Run the cache test script:

```bash
python test_cache.py
```

Expected output:
```
Performance Comparison:
  First query:  21.45 seconds (full processing)
  Second query: 0.00 seconds (cached)
  Speedup:      16657.1x faster
```

## Cache Validation

The system includes automatic validation:
- Expired entries are automatically removed
- Cache integrity is checked before returning results
- Age metadata is included in cached results

## Use Cases

### High-Value Scenarios

1. **Repeated Queries**
   - Users frequently looking up the same common terms
   - Example: "diabetes", "hypertension", "CBC"

2. **Testing & Development**
   - Rapid iteration without API costs
   - Instant feedback for UI/UX testing

3. **Demonstration**
   - Show system capabilities without delays
   - Consistent response times

4. **API Cost Optimization**
   - Reduce OpenAI API calls for common queries
   - Decrease Clinical Tables API load

### When Cache Is NOT Used

1. **Continuation Requests**
   - "more", "next", "show more" use pagination logic, not cache
   - Ensures correct page sequencing

2. **Different Queries**
   - Even similar terms are treated as distinct
   - "diabetes" ≠ "diabetes mellitus" ≠ "type 2 diabetes"

3. **After Cache Clear**
   - Manual `clear cache` command resets all cached queries

## Monitoring

### Cache Hit Logging

Cached queries log clearly:
```
INFO - Retrieved cached results for query 'diabetes' (age: 5s)
INFO - Returning cached results for: diabetes
```

Fresh queries log:
```
INFO - Looking up: diabetes
INFO - Starting agentic workflow for query: diabetes
```

### Cache Metrics

Available in `get_summary()`:
```python
{
    "has_memory": True,
    "cache_size": 1,
    "cache_enabled": True,
    "last_query": "diabetes",
    ...
}
```

## Best Practices

### For Users

1. ✅ **Use exact same query text** for cache hits
2. ✅ **Check cache status** periodically to understand system behavior
3. ✅ **Clear cache** when testing new features or updated medical data
4. ❌ Don't rely on cache for time-sensitive medical updates

### For Developers

1. ✅ **Test cache invalidation** after code changes
2. ✅ **Monitor cache hit rates** in production
3. ✅ **Adjust TTL** based on use case (longer for stable terms, shorter for volatile data)
4. ✅ **Consider cache warming** for common terms on startup

## Limitations

1. **Memory Usage**: Cache is in-memory (not persistent across restarts)
2. **No Persistence**: Caches clear when application stops
3. **No Distributed Cache**: Each instance has separate cache
4. **Fixed Capacity**: Max 100 queries (LRU eviction)

## Future Enhancements

Potential improvements:
- [ ] Redis/Memcached for distributed caching
- [ ] Persistent cache (save to disk)
- [ ] Configurable TTL per query or term type
- [ ] Cache warming from common query lists
- [ ] Cache analytics dashboard
- [ ] Selective cache invalidation (by dataset, term type, etc.)

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md) - Cache testing procedures
- [Configuration Guide](../README.md) - System configuration
- [Memory & State](../README.md) - Pagination and memory features
