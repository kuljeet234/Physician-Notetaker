# Medical Transcription NLP Pipeline

A Python-based system for extracting medical information from doctor-patient conversations. It automatically identifies symptoms, diagnoses, treatments, and generates structured SOAP notes.

## What It Does

- Extracts symptoms, diagnoses, and treatments from medical transcripts
- Analyzes patient sentiment (anxious, reassured, neutral)
- Generates structured SOAP notes automatically
- Works with any type of medical conversation (accidents, chronic conditions, emergencies)

## Requirements

Python 3.8 or higher

## Installation

Install the required packages:
```bash
pip install torch transformers sentence-transformers scikit-learn spacy
```

Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the script:
```bash
python medical_nlp_pipeline.py
```

The script will process the sample transcript and output three main sections:
1. Medical Summary (patient name, symptoms, diagnosis, treatment, etc.)
2. Sentiment Analysis (patient's emotional state and intent)
3. SOAP Note (structured clinical documentation)

Results are saved to `clinical_results.json`

## How to Use Your Own Transcript

Replace the `SAMPLE` variable in the main section with your transcript. Format it like this:
```
Physician: How are you feeling?
Patient: I have chest pain.
Physician: When did it start?
Patient: This morning.
```

The script handles different medical scenarios automatically.

## Output Format

The pipeline generates JSON output with three sections:

**Medical Summary**: Patient info, symptoms list, diagnosis, treatments, current status, and prognosis

**Sentiment Analysis**: Patient's emotional state and their intent (seeking reassurance, reporting symptoms, etc.)

**SOAP Note**: Clinical documentation with Subjective, Objective, Assessment, and Plan sections

## Models Used

- BioBERT for medical entity recognition
- RoBERTa for question-answering
- BART for zero-shot classification
- Sentence Transformers for semantic validation
- spaCy for text processing

## Notes

First run will be slower because it downloads the models. Subsequent runs are much faster.

GPU is recommended but not required. The script automatically uses GPU if available.

The validation system filters out incorrect extractions, so you might see fewer results than expected - this is intentional to maintain accuracy.

## Limitations

- Works best with clear, structured conversations
- May miss very informal or abbreviated medical terms
- Requires complete sentences for best results
- SOAP note quality depends on how much information is in the transcript

## Troubleshooting

**Import errors**: Make sure all packages are installed with the correct versions

**Memory issues**: Try setting `use_gpu=False` when initializing the pipeline

**spaCy model not found**: Run the download command again

**Slow performance**: First run downloads models. If still slow, your transcript might be very long - consider splitting it into sections