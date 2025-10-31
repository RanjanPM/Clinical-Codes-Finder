"""
Medical Terminology Resolution Agent
Uses LLM to identify the type of medical term and map to appropriate coding systems
"""

import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import config

logger = logging.getLogger(__name__)


class TerminologyAgent:
    """Agent for resolving medical terminology and identifying appropriate coding systems"""
    
    # Mapping of term types to relevant datasets
    TERM_TYPE_DATASETS = {
        "diagnosis": ["icd10cm", "icd11", "icd9cm_dx", "conditions"],
        "procedure": ["hcpcs", "icd9cm_sg", "procedures"],
        "lab_test": ["loinc"],
        "medication": ["rxterms", "drugs"],
        "medical_equipment": ["hcpcs"],  # DME, prosthetics, wheelchairs, etc.
        "unit": ["ucum"],
        "phenotype": ["hpo", "icd10cm", "conditions"],  # Include ICD codes for phenotypes too
        "genetic_variant": ["clinvar", "snps"],
        "gene": ["genes"],
        "genetic_disease": ["genetic_diseases", "hpo"],  # Include HPO for genetic diseases
        "pharmacogenomics": ["pharmvar"],
        "provider": ["npi_idv", "npi_org"]
    }
    
    SYSTEM_PROMPT = """You are a medical terminology expert. Your job is to analyze clinical terms and identify:
1. The type of medical term (diagnosis, procedure, lab test, medication, unit, phenotype, genetic variant, gene, provider, etc.)
2. Which medical coding systems would be most relevant

Available term types:
- diagnosis: For diseases, conditions, symptoms (ICD-10-CM, ICD-11, ICD-9-CM) - use for general clinical diagnoses
- procedure: For medical procedures, surgeries, treatments (HCPCS, ICD-9-CM procedures)
- lab_test: For laboratory tests and panels (LOINC)
- medication: For drugs and medications (RxTerms)
- medical_equipment: For durable medical equipment (DME), prosthetics, orthotics, mobility aids, assistive devices (HCPCS).
  Use this for terms like: wheelchair, walker, crutches, prosthetic limb, oxygen equipment, hospital bed, nebulizer, etc.
  HCPCS codes are used for billing and reimbursement of medical equipment.
- unit: For units of measure (UCUM) - e.g., mg, mL, mmol/L
- phenotype: For observable characteristics, clinical features, or traits used in genetic/genomic contexts (HPO). 
  Use this for terms like: ataxia, seizures, intellectual disability, dysmorphic features, developmental delay, etc.
  HPO is especially important for hereditary conditions, congenital abnormalities, and genetic syndromes.
- genetic_variant: For genetic variants (ClinVar, SNPs)
- gene: For gene names (HGNC/NCBI Genes)
- genetic_disease: For hereditary/genetic diseases
- pharmacogenomics: For drug-gene interactions (PharmVar)
- provider: For healthcare providers (NPI)

IMPORTANT DECISION CRITERIA:
- If a term describes a clinical feature that could be part of a genetic syndrome or hereditary condition, classify it as "phenotype"
- If a term is commonly used in genetics or has a known HPO code, classify it as "phenotype"
- If a term refers to equipment, devices, or assistive aids (not medications), classify it as "medical_equipment"
- Examples of phenotype terms: ataxia, seizures, hypotonia, intellectual disability, dysmorphic facies, polydactyly
- Examples of medical equipment: wheelchair, walker, crutches, CPAP machine, prosthetic, oxygen concentrator

Respond in JSON format with:
{
    "term_type": "diagnosis|procedure|lab_test|medication|phenotype|...",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "search_terms": ["alternative search terms"],
    "primary_datasets": ["most relevant datasets"],
    "secondary_datasets": ["additional relevant datasets"]
}"""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        self.llm = ChatOpenAI(
            model=model_name or config.llm.TERMINOLOGY_MODEL,
            temperature=temperature if temperature is not None else config.llm.TERMINOLOGY_TEMPERATURE
        )
    
    def analyze_term(self, term: str) -> Dict[str, Any]:
        """
        Analyze a medical term and determine its type and relevant datasets
        
        Args:
            term: The clinical term to analyze
            
        Returns:
            Dictionary with term analysis
        """
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"Analyze this medical term: '{term}'")
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                    
                result = json.loads(content)
                
                # Validate and enhance result
                if "term_type" not in result:
                    result["term_type"] = "diagnosis"  # Default fallback
                
                # Always use our predefined dataset mappings based on term_type
                # Override whatever the LLM suggested to ensure correct dataset names
                result["primary_datasets"] = self.TERM_TYPE_DATASETS.get(
                    result["term_type"], 
                    ["icd10cm", "loinc", "rxterms"]
                )
                
                if "search_terms" not in result:
                    result["search_terms"] = [term]
                
                logger.info(f"Term analysis for '{term}': {result['term_type']} (confidence: {result.get('confidence', 0)})")
                logger.info(f"Primary datasets: {result.get('primary_datasets')}")
                logger.info(f"Search terms: {result.get('search_terms')}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # Fallback: simple term type detection
                return self._fallback_analysis(term)
                
        except Exception as e:
            logger.error(f"Error analyzing term '{term}': {e}")
            return self._fallback_analysis(term)
    
    def _fallback_analysis(self, term: str) -> Dict[str, Any]:
        """Fallback analysis using keyword matching"""
        term_lower = term.lower()
        
        # Simple keyword-based detection
        if any(kw in term_lower for kw in ["test", "panel", "assay", "glucose", "hemoglobin", "blood"]):
            term_type = "lab_test"
        elif any(kw in term_lower for kw in ["mg", "tablet", "capsule", "drug", "medication"]):
            term_type = "medication"
        elif any(kw in term_lower for kw in ["wheelchair", "walker", "crutch", "prosthetic", "orthotic", "cpap", "oxygen", "nebulizer", "hospital bed", "mobility device", "assistive device", "dme"]):
            term_type = "medical_equipment"
        elif any(kw in term_lower for kw in ["surgery", "procedure", "operation", "therapy"]):
            term_type = "procedure"
        elif any(kw in term_lower for kw in ["gene", "dna", "variant", "mutation"]):
            term_type = "genetic_variant"
        else:
            term_type = "diagnosis"
        
        return {
            "term_type": term_type,
            "confidence": 0.5,
            "reasoning": "Fallback keyword-based detection",
            "search_terms": [term],
            "primary_datasets": self.TERM_TYPE_DATASETS.get(term_type, ["icd10cm"]),
            "secondary_datasets": []
        }
