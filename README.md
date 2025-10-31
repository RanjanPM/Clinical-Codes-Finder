# Clinical Term Lookup System - Agentic RAG

An intelligent agentic RAG (Retrieval-Augmented Generation) system for clinical data lookup using LangChain/LangGraph and Clinical Tables APIs.

## 🚀 Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the system
python main.py "diabetes"

# Or interactive mode
python main.py
```

## 📁 Project Structure

```
CVS/
├── agents/               # AI agents (terminology, retrieval, scoring, synthesis)
├── apis/                 # Clinical Tables API clients
├── config.py             # Configuration management
├── docs/                 # Documentation
│   ├── README.md         # Main documentation
│   ├── CACHING.md        # Query caching guide
│   ├── PAGINATION_GUIDE.md
│   ├── SCORING_GUIDE.md
│   ├── TESTING_GUIDE.md
│   └── PRODUCTION_GUIDE.md
├── graph/                # LangGraph workflow orchestration
├── memory/               # Conversation memory and state management
├── scripts/              # Utility scripts
│   ├── lookup.py         # CLI wrapper
│   ├── lookup.ps1        # PowerShell wrapper
│   └── run_tests.py      # Test runner
├── tests/                # Test suite
│   ├── test_units.py     # Unit tests
│   ├── test_integration.py
│   ├── demo_cache.py     # Caching demo
│   └── demo_pagination.py
├── main.py               # Main application
├── requirements.txt      # Python dependencies
└── .env.example          # Environment template
```

## 🎯 Key Features

### Agentic Architecture
- **Multi-Agent System**: Specialized agents for terminology, retrieval, scoring, and synthesis
- **LangGraph Orchestration**: Intelligent workflow management with conditional routing
- **Iterative Refinement**: Automatic quality checks and re-querying if needed

### Medical Data Coverage
- **15+ Medical Coding Systems**: ICD-10-CM, ICD-11, ICD-9-CM, LOINC, RxTerms, HCPCS, UCUM, HPO
- **Genomics Support**: ClinVar variants, COSMIC mutations, dbVar, SNPs, Genes (HGNC/NCBI)
- **Provider Data**: NPI lookups for individuals and organizations
- **Pharmacogenomics**: PharmVar star alleles, drug-gene interactions

### Performance Features
- **Query Result Caching**: ~16,000x faster for repeated queries
- **Pagination Support**: Handle large result sets efficiently
- **Smart Scoring**: Configurable relevance scoring with optional LLM enhancement
- **API Caching**: Reduced API calls through intelligent caching

### User Experience
- **Interactive Mode**: Natural conversation-based queries
- **Single Command**: Quick lookups with `python main.py "term"`
- **Pagination**: Simple "more" or "next" to see additional results
- **Cache Management**: Built-in cache status and clearing

## 📖 Documentation

See the `docs/` folder for comprehensive documentation:

- **[docs/README.md](docs/README.md)** - Full system documentation
- **[docs/CACHING.md](docs/CACHING.md)** - Query caching guide


## 🧪 Testing

```powershell
# Run all tests
python scripts/run_tests.py

# Run specific tests
pytest tests/test_units.py -v
pytest tests/test_integration.py -v

```

## 🔧 Configuration

Edit `.env` file to customize:

```env
# LLM Configuration
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Pagination
RESULTS_PER_PAGE=10
MAX_TOTAL_RESULTS=100

# Caching
CLINICAL_API_CACHE_TTL=3600
QUERY_CACHE_TTL=3600

# Quality Control
QUALITY_THRESHOLD=0.60
MAX_ITERATIONS=2
```

See `.env.example` for all 40+ configuration options.

## 💡 Usage Examples

### Single Query
```powershell
python main.py "type 2 diabetes"
python main.py "hemoglobin A1c test"
python main.py "lisinopril"
```

### Interactive Mode
```powershell
python main.py

Enter clinical term: diabetes
[Results displayed]

Enter clinical term: more
[Next page of results]

Enter clinical term: cache status
Cache: 1 queries cached

Enter clinical term: clear cache
Cache cleared!
```

### Using the Wrapper Script
```powershell
.\scripts\lookup.ps1 diabetes
.\scripts\lookup.ps1 "blood pressure medication"
```

## 🏥 Clinical Specialties

The system handles queries across multiple medical specialties:

- **Cardiology**: Heart conditions, ECG codes, cardiac risk assessment
- **Oncology**: Cancer genomics, tumor classifications, treatment protocols
- **Pharmacogenomics**: Drug-gene interactions, dosing recommendations
- **Diagnostics**: Lab values (LOINC), reference ranges
- **Genomics**: Variant analysis, hereditary disease risk
- **Provider Lookup**: NPI-based provider search

## 🚀 Performance

- **First Query**: 15-25 seconds (full agent processing)
- **Cached Query**: <0.01 seconds (~16,000x faster)
- **Pagination**: Instant (no re-processing)
- **API Efficiency**: Intelligent caching reduces redundant calls

## 🔒 Compliance & Safety

- **HIPAA Considerations**: No PHI storage, encrypted transmission
- **Medical Liability**: Clear disclaimers, human oversight required
- **Audit Trail**: Comprehensive logging for compliance
- **Data Privacy**: No personal health information in logs

## 🛠️ Development

```powershell
# Install dev dependencies
pip install -r requirements-test.txt

# Run tests with coverage
pytest --cov=. tests/

# Check code style
flake8 agents/ apis/ graph/ memory/

# Format code
black agents/ apis/ graph/ memory/
```

## 📊 Architecture

```
User Query
    ↓
[Terminology Agent] → Analyze query type & extract search terms
    ↓
[Retrieval Agent] → Fetch codes from 15+ medical databases
    ↓
[Scoring Agent] → Score relevance (rule-based + optional LLM)
    ↓
[Quality Check] → Meets threshold? Yes → Synthesize, No → Refine & retry
    ↓
[Synthesis Agent] → Generate structured response with insights
    ↓
[Cache Results] → Store for instant future retrieval
    ↓
User Response
```

## 🤝 Contributing

1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## 📝 License

This project is for educational and research purposes. Always consult qualified medical professionals for clinical decisions.

## 🆘 Support

- Check `docs/` folder for detailed guides
- Review test files in `tests/` for usage examples


## ⚡ Quick Commands Reference

```powershell
# Single lookup
python main.py "diabetes"

# Interactive mode
python main.py

# Run tests
python scripts/run_tests.py

# Demo caching (see 16,000x speedup!)
python tests/demo_cache.py

# Demo pagination
python tests/demo_pagination.py

# Check system
python tests/test_system.py
```

---

**Note**: This is a clinical decision support tool. Always verify results with official medical coding resources and qualified healthcare professionals.
