# Clinical Term Lookup System

An intelligent agentic RAG system that finds relevant medical codes across multiple coding systems for any clinical term using autonomous agents, iterative refinement, and quality-driven search.

## Features

### Core Capabilities
- **Intelligent Term Analysis**: Uses GPT-4o-mini to identify medical term types and select relevant datasets
- **Multi-System Search**: Searches across 15+ medical coding systems in parallel
- **Agentic Workflow**: Autonomous agents collaborate through LangGraph for optimal results
- **Iterative Refinement**: Automatically refines searches up to 3-6 iterations based on quality thresholds
- **Quality Scoring**: Multi-factor relevance scoring (text similarity, dataset appropriateness, code specificity)
- **Intelligent Synthesis**: LLM-powered summaries with clinical context and recommendations
- **Fast & Cached**: Built-in caching for improved performance with configurable TTLs

### Supported Medical Coding Systems
- **Diagnoses**: ICD-10-CM, ICD-11, ICD-9-CM (diagnoses)
- **Lab Tests**: LOINC (laboratory observations)
- **Medications**: RxTerms (prescribable drugs)
- **Procedures**: HCPCS, ICD-9-CM (procedures)
- **Genomics**: ClinVar variants, COSMIC mutations, dbVar, Genes (HGNC/NCBI), SNPs
- **Clinical**: HPO (human phenotypes), Medical Conditions
- **Pharmacogenomics**: PharmVar (star alleles)
- **Providers**: NPI (healthcare providers)

## Installation

1. **Clone and setup environment:**
```powershell
cd c:\Code\CVS
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Configure API key:**
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Interactive Mode
```powershell
python main.py
```

Then enter clinical terms when prompted:
```
Enter clinical term: blood sugar test
Enter clinical term: tuberculosis
Enter clinical term: metformin 500 mg
```

### Single Lookup
```powershell
python main.py "blood sugar test"
```

### Quick Lookup Commands
```powershell
# Using wrapper scripts (simpler)
.\lookup diabetes
.\lookup "blood sugar test"
lookup metformin

# Direct Python execution
python main.py diabetes
```

### Query Result Caching
The system automatically caches query results for **1 hour** to provide instant responses:

```
Enter clinical term: diabetes
Searching medical codes for 'diabetes'...
... (20 seconds processing) ...

Enter clinical term: diabetes
Retrieving cached results for 'diabetes'...
... (instant response - <0.01 seconds!) ...

Query: diabetes
Source: Cached results (age: 5.2 seconds)
Note: No API calls or LLM processing - instant retrieval
```

**Cache commands:**
- `clear cache` - Clear all cached queries
- `cache status` - View cache statistics

**Performance benefits:**
- ✅ No LLM API calls (saves OpenAI costs)
- ✅ No agent processing required
- ✅ Instant response (~16,000x faster)
- ✅ 1-hour TTL with auto-cleanup

See [Caching Documentation](docs/CACHING.md) for details.

### Pagination - View More Results
When a query returns many codes, only the first page is shown (default: 10 codes per system).
To see more results, simply type:
```
Enter clinical term: diabetes
... (showing 1-10 of 37 results) ...

TIP: More results available!
   Type 'more', 'next', or 'show more' to see additional codes.

Enter clinical term: more
... (showing 11-20 of 37 results) ...

Enter clinical term: next
... (showing 21-30 of 37 results) ...
```

**Supported continuation keywords:**
- `more`, `next`, `continue`
- `show more`, `see more`, `keep going`
- `more results`, `additional`, `rest`

The system remembers your last query and automatically shows the next page of results.

### Test the System
```powershell
python test_integraton.py
python test_integration_pagination.py  # Test pagination features
```

## Example Output

```
================================================================================
Query: diabetes
Term Type: diagnosis
Confidence: 100.00%
Reasoning: The term 'diabetes' is a well-known endocrine disorder affecting glucose metabolism
================================================================================

AGENTIC WORKFLOW METRICS
--------------------------------------------------------------------------------
  Iterations Performed: 1
  Result Quality Score: 60.97%
  Total Matches: 48
  Average Relevance: 44.00%
  High Quality Results: 15

DETAILED CODES (2 coding system(s))
================================================================================

ICD-10-CM (35 results)
--------------------------------------------------------------------------------
  1. [HIGH] Code: E11.9 (Relevance: 0.87)
     Description: Type 2 diabetes mellitus without complications
     
  2. [HIGH] Code: E10.9 (Relevance: 0.85)
     Description: Type 1 diabetes mellitus without complications
     
  3. [MEDIUM] Code: E11.65 (Relevance: 0.72)
     Description: Type 2 diabetes mellitus with hyperglycemia

Medical Conditions (13 results)
--------------------------------------------------------------------------------
  1. [HIGH] Code: C0011849 (Relevance: 0.91)
     Description: Diabetes mellitus
     
  2. [MEDIUM] Code: C0011854 (Relevance: 0.68)
     Description: Type 1 diabetes mellitus

================================================================================
INTELLIGENT SYNTHESIS
================================================================================

EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Found 48 highly relevant codes for 'diabetes' across 2 coding systems.
Primary classification: Type 2 diabetes mellitus (E11.9 in ICD-10-CM).
High confidence match with strong clinical support.

SEARCH QUALITY: [GOOD]
   Quality threshold met on first iteration with 60.97% score.

KEY PATTERNS
--------------------------------------------------------------------------------
  - Most results classify diabetes by type (Type 1, Type 2, Gestational)
  - Complications coded separately in ICD-10-CM
  - LOINC codes available for glucose monitoring

TOP RECOMMENDATIONS
--------------------------------------------------------------------------------
  1. [ICD-10-CM] E11.9
     Use Case: General Type 2 diabetes diagnosis without specified complications
     Confidence: [HIGH CONFIDENCE]
     
  2. [ICD-10-CM] E10.9
     Use Case: Type 1 diabetes diagnosis without specified complications
     Confidence: [HIGH CONFIDENCE]
```

## Architecture

### Agentic Components

1. **Terminology Agent** (`agents/terminology_agent.py`)
   - Analyzes medical terms using LLM (GPT-4o-mini)
   - Determines term type (diagnosis, medication, lab test, genetic, etc.)
   - Maps terms to relevant coding systems dynamically
   - Generates alternative search terms for refinement

2. **Retrieval Agent** (`agents/retrieval_agent.py`)
   - Executes parallel async searches across selected datasets
   - Handles multiple search term variations
   - Merges and deduplicates results across iterations

3. **Scoring Agent** (`agents/scoring_agent.py`)
   - Multi-factor relevance scoring algorithm:
     * Text similarity (30%)
     * Dataset appropriateness (20%)
     * Code specificity (15%)
     * Description quality (10%)
     * Query term presence (25%)
   - Categorizes results: HIGH, MEDIUM, LOW, VERY LOW

4. **Refinement Agent** (`agents/refinement_agent.py`)
   - Evaluates result quality and count
   - Generates refinement strategies (broaden, narrow, alternative)
   - Prevents search loops with history tracking
   - Adapts approach based on iteration count

5. **Synthesis Agent** (`agents/synthesis_agent.py`)
   - Generates executive summaries using LLM
   - Identifies key patterns across results
   - Provides top recommendations with use cases
   - Assesses search quality and confidence

6. **Clinical Workflow** (`graph/clinical_workflow.py`)
   - LangGraph-based iterative orchestration
   - Conditional routing based on quality evaluation
   - Early stopping when excellent results found
   - Maintains state across iterations

### Agentic Workflow

```
User Query
    ↓
[Terminology Agent] → Analyze term type & select datasets
    ↓
┌─→ [Retrieval Agent] → Search selected datasets (parallel)
│   ↓
│   [Scoring Agent] → Score results for relevance
│   ↓
│   [Evaluate Quality] → Check if threshold met
│   ↓
│   ├─ Quality Good? → [Synthesis Agent] → Final Response
│   └─ Quality Low? → [Refinement Agent] → New strategy
│       └─────────────────┘ (iterate up to MAX_ITERATIONS)
```

### Key Features

- **Dynamic Dataset Selection**: Only queries relevant APIs based on term analysis
- **Iterative Refinement**: Automatically refines search if quality < threshold
- **Quality-Driven**: Stops early if excellent results found (quality >= 0.8)
- **State Management**: Tracks search history, iterations, and quality metrics
- **Configurable**: All thresholds and weights adjustable via environment variables

## Supported Medical Term Types

| Term Type | Coding Systems | Examples |
|-----------|---------------|----------|
| Diagnosis | ICD-10-CM, ICD-11, ICD-9 | "tuberculosis", "diabetes" |
| Lab Test | LOINC | "blood sugar test", "hemoglobin" |
| Medication | RxTerms | "metformin 500 mg", "aspirin" |
| Procedure | HCPCS, ICD-9-SG | "appendectomy", "MRI scan" |
| Genetic Variant | ClinVar, SNPs | "BRCA1 mutation" |
| Gene | HGNC, NCBI Genes | "TP53", "CFTR" |
| Provider | NPI | "cardiologist", "hospital" |

## Configuration

The system is fully configurable via environment variables. Copy `.env.example` to `.env` and customize:

### Essential Configuration
```bash
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here
```

### Agentic Workflow Configuration
```bash
# Iteration settings
MAX_ITERATIONS=3                    # Maximum refinement iterations
MIN_QUALITY_THRESHOLD=0.6           # Minimum quality score to accept
MIN_RESULTS_THRESHOLD=3             # Minimum results needed
ENABLE_EARLY_STOPPING=true          # Stop early if excellent results
EXCELLENT_QUALITY_THRESHOLD=0.8     # Quality score for early stopping

# Quality calculation weights
QUALITY_RELEVANCE_WEIGHT=0.7        # Weight for average relevance (70%)
QUALITY_COUNT_WEIGHT=0.3            # Weight for result count (30%)
```

### Pagination Configuration
```bash
# Display settings
MAX_CODES_PER_SYSTEM=10            # Number of codes shown per page per system
```

When more results are available than MAX_CODES_PER_SYSTEM, the system displays a pagination hint. Users can type continuation keywords ('more', 'next', 'show more') to view additional pages.

### Result Scoring Configuration
```bash
# Multi-factor scoring weights (must sum to 1.0)
TEXT_SIMILARITY_WEIGHT=0.30
DATASET_APPROPRIATENESS_WEIGHT=0.20
CODE_SPECIFICITY_WEIGHT=0.15
DESCRIPTION_QUALITY_WEIGHT=0.10
QUERY_TERM_PRESENCE_WEIGHT=0.25

# Relevance thresholds
HIGH_RELEVANCE_THRESHOLD=0.8        # Score >= 0.8 is HIGH
MEDIUM_RELEVANCE_THRESHOLD=0.6      # Score >= 0.6 is MEDIUM
LOW_RELEVANCE_THRESHOLD=0.4         # Score >= 0.4 is LOW
```

### Search Refinement Configuration
```bash
TOO_MANY_RESULTS_THRESHOLD=50       # Narrow search if more results
TOO_FEW_RESULTS_THRESHOLD=1         # Broaden search if fewer results
NUM_ALTERNATIVE_TERMS=5             # Alternative terms to generate
ALTERNATIVE_STRATEGY_AFTER_ITERATIONS=2  # Try alternatives after N iterations
```

### LLM Configuration
```bash
# Model selection for each agent
TERMINOLOGY_MODEL=gpt-4o-mini
REFINEMENT_MODEL=gpt-4o-mini
SCORING_MODEL=gpt-4o-mini
SYNTHESIS_MODEL=gpt-4o-mini

# Temperature settings
TERMINOLOGY_TEMPERATURE=0.1
REFINEMENT_TEMPERATURE=0.3
SCORING_TEMPERATURE=0.1
SYNTHESIS_TEMPERATURE=0.2
```

### API Configuration
```bash
CLINICAL_TABLES_BASE_URL=https://clinicaltables.nlm.nih.gov/api
CLINICAL_TABLES_RATE_LIMIT=100
MAX_RESULTS_PER_DATASET=10
CACHE_TTL_STABLE_CODES=86400        # 24 hours
CACHE_TTL_DYNAMIC_DATA=3600         # 1 hour
CACHE_MAX_SIZE=10000
```

### Display Configuration
```bash
MAX_CODES_PER_SYSTEM=5              # Codes to display per system
MAX_TOP_RECOMMENDATIONS=10          # Top recommendations in synthesis
SHOW_RELEVANCE_SCORES=true          # Show relevance scores
SHOW_ITERATION_HISTORY=true         # Show iteration details
SHOW_SYNTHESIS=true                 # Show synthesis section
SHOW_QUALITY_METRICS=true           # Show quality metrics
LOG_LEVEL=INFO                      # Logging level
```

## Error Handling

The system implements robust error handling:
- API failures with fallback mechanisms
- Medical code validation
- LLM response parsing with fallbacks
- Comprehensive logging

## Performance Optimizations

- **Intelligent Caching**: Configurable TTLs (24h for stable codes, 1h for dynamic data)
- **Parallel Execution**: Async searches across multiple datasets simultaneously
- **Early Stopping**: Terminates iteration when excellent quality achieved (>= 0.8)
- **Rate Limiting**: Respects API limits to prevent throttling
- **Memory Efficient**: Streaming results for large datasets
- **Quality Thresholds**: Avoids unnecessary iterations when results sufficient

## Testing

### Full System Test
```powershell
python test_integration.py
```
Tests complete agentic workflow with LLM integration (requires OPENAI_API_KEY).

## Dependencies

- **LangChain 0.3.13**: LLM integration and prompting
- **LangGraph 0.2.52**: Multi-agent workflow orchestration
- **OpenAI 1.57.4**: GPT-4o-mini for term analysis and synthesis
- **aiohttp 3.11.11**: Async HTTP client for parallel API calls
- **Clinical Tables API**: Medical code database (free, no authentication)
- **python-dotenv**: Environment variable management
- **cachetools**: In-memory caching with TTL

## System Requirements

- Python 3.10+
- OpenAI API key
- Internet connection for Clinical Tables API
- 2GB RAM minimum (4GB recommended for large queries)

## Agentic RAG Compliance

This system implements a complete Agentic RAG architecture:

1. **Planning**: Terminology agent infers which APIs to query based on term analysis
2. **Tooling**: Dynamic dataset selection - no static pre-search
3. **Memory/State**: Full iteration history and search state management
4. **Selection**: Multi-factor quality scoring filters noise (only >= 60% relevance)
5. **Attribution**: Results grouped by coding system with full metadata
6. **Summary**: LLM-generated synthesis with traceable reasoning

## Project Structure

```
CVS/
├── agents/                      # Autonomous agents
│   ├── terminology_agent.py     # Term analysis & dataset selection
│   ├── retrieval_agent.py       # Multi-dataset search
│   ├── refinement_agent.py      # Search strategy refinement
│   ├── scoring_agent.py         # Relevance scoring
│   └── synthesis_agent.py       # Intelligent summarization
├── apis/
│   └── clinical_tables.py       # Clinical Tables API client
├── graph/
│   └── clinical_workflow.py     # LangGraph workflow orchestration
├── config.py                    # Centralized configuration
├── main.py                      # Main application
├── .env.example                 # Configuration template
└── requirements.txt             # Python dependencies
```

## Troubleshooting

**Issue**: "OPENAI_API_KEY not found"
- Solution: Create `.env` file with your OpenAI API key

**Issue**: HTTP 404 errors for ICD-11
- Solution: ICD-11 endpoint may be unavailable; system continues with other datasets

**Issue**: Low quality scores
- Solution: Adjust `MIN_QUALITY_THRESHOLD` in `.env` or let system iterate

**Issue**: Too many iterations
- Solution: Increase `MAX_ITERATIONS` or lower quality thresholds in `.env`

## Contributing

Contributions welcome! Please open an issue or pull request.

## Support

For questions or issues, please open a GitHub issue with:
- Query that caused the issue
- Log output
- Configuration settings (without API keys)
