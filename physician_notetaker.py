"""Clinical transcription pipeline extracted from
NLP_Physician_Notetaker.ipynb so it can be imported, tested,
and run as a script.

Usage:
    python physician_notetaker.py            # runs the bundled SAMPLE
    from physician_notetaker import ClinicalTranscriptionPipeline
"""

import re
import json
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MedicalSummary:
    Patient_Name: str
    Symptoms: List[str]
    Diagnosis: str
    Treatment: List[str]
    Current_Status: str
    Prognosis: str
    Confidence_Score: float


@dataclass
class SentimentResult:
    Sentiment: str
    Intent: str
    Confidence: float


@dataclass
class SOAPNote:
    Subjective: Dict[str, str]
    Objective: Dict[str, str]
    Assessment: Dict[str, str]
    Plan: Dict[str, str]


class MedicalEntityValidator:
    def __init__(self, sentence_model):
        self.sentence_model = sentence_model

        self.valid_examples = {
            'symptom': [
                "neck pain", "back pain", "headache", "nausea", "dizziness", "fatigue",
                "chest pain", "weakness", "numbness", "tingling", "swelling", "stiffness",
                "fever", "cough", "sore throat", "abdominal pain", "joint pain", "muscle aches",
                "blurred vision", "loss of appetite", "weight loss", "difficulty breathing",
                "palpitations", "confusion", "increased thirst", "frequent urination",
                "discomfort", "ache", "soreness", "tenderness", "bruising", "rash"
            ],
            'diagnosis': [
                "whiplash injury", "diabetes", "hypertension", "pneumonia", "fracture",
                "sprain", "strain", "arthritis", "bronchitis", "asthma", "heart disease",
                "stroke", "infection", "inflammation", "herniated disc", "concussion",
                "type 2 diabetes", "migraine", "gastritis", "contusion", "laceration"
            ],
            'treatment': [
                "physiotherapy", "physical therapy", "surgery", "rehabilitation",
                "exercise program", "massage therapy", "rest", "ice therapy", "heat therapy",
                "occupational therapy", "dietary modifications", "lifestyle changes",
                "blood sugar monitoring", "insulin therapy", "medication management",
                "pain management", "wound care", "immobilization", "compression therapy"
            ],
            'medication': [
                "ibuprofen", "aspirin", "acetaminophen", "painkiller", "antibiotic",
                "metformin", "insulin", "anti-inflammatory", "analgesic", "steroid"
            ]
        }

        self.valid_embeddings = {}
        for entity_type, examples in self.valid_examples.items():
            embeddings = sentence_model.encode(examples)
            self.valid_embeddings[entity_type] = embeddings

        self.forbidden_phrases = [
            'these symptoms', 'your symptoms', 'those symptoms', 'the symptoms',
            'this symptom', 'that symptom', 'my symptoms', 'any symptoms',
            'such symptoms', 'similar symptoms', 'following symptoms',
            'some discomfort', 'any discomfort', 'the discomfort',
            'any pain', 'the pain', 'some pain', 'this pain', 'that pain',
            'any lingering pain', 'lingering pain', 'the stiffness', 'some stiffness',
            'pain now', 'occasional backaches', 'occasional backache',
            'experiencing pain', 'feel pain', 'take pain', 'felt pain',
            'what', 'when', 'where', 'why', 'how', 'which', 'who',
            'feeling', 'noticed', 'started', 'stopped', 'began',
            'help with the', 'to help', 'physiotherapy to'
        ]

        self.single_word_forbidden = [
            'pain', 'discomfort', 'stiffness', 'ache', 'neck', 'back',
            'head', 'help', 'with', 'the', 'experiencing', 'feel', 'take'
        ]

    def is_valid_entity(self, text: str, entity_type: str) -> Tuple[bool, float]:
        text_lower = text.lower().strip()

        if len(text_lower) < 3 or len(text_lower) > 60:
            return False, 0.0

        if text_lower in self.forbidden_phrases:
            return False, 0.0

        for forbidden in ['to help with', 'help with the', 'occasional']:
            if forbidden in text_lower:
                return False, 0.0

        words = text_lower.split()
        if len(words) == 1 and text_lower in self.single_word_forbidden:
            return False, 0.0

        if words[0] in ['what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'occasional']:
            return False, 0.0

        filler_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'their', 'some', 'any'}
        content_words = [w for w in words if w not in filler_words]
        if len(content_words) == 0:
            return False, 0.0

        if entity_type == 'symptom':
            if len(words) <= 2 and not any(kw in text_lower for kw in ['pain', 'ache', 'discomfort', 'soreness', 'weakness', 'numbness', 'tingling', 'swelling', 'stiffness', 'backache', 'headache']):
                return False, 0.0

        if entity_type in self.valid_embeddings:
            text_embedding = self.sentence_model.encode([text_lower])[0]
            similarities = cosine_similarity(
                text_embedding.reshape(1, -1),
                self.valid_embeddings[entity_type]
            )[0]
            max_similarity = float(np.max(similarities))

            thresholds = {
                'symptom': 0.58,
                'diagnosis': 0.60,
                'treatment': 0.55,
                'medication': 0.65
            }

            threshold = thresholds.get(entity_type, 0.60)

            if max_similarity < threshold:
                return False, max_similarity

            return True, max_similarity

        return True, 0.5


class EnhancedClinicalExtractor:
    def __init__(self, use_gpu: bool = True):
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1

        print("Loading models...")

        try:
            self.medical_ner = pipeline(
                "ner",
                model="alvaroalon2/biobert_diseases_ner",
                aggregation_strategy="max",
                device=self.device
            )
        except:
            self.medical_ner = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="max",
                device=self.device
            )

        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=self.device
        )

        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )

        try:
            self.nlp = spacy.load("en_core_web_trf")
        except:
            self.nlp = spacy.load("en_core_web_sm")

        self.validator = MedicalEntityValidator(self.sentence_model)
        print("✓ Models loaded\n")

    def extract_symptoms_robust(self, text: str) -> Set[str]:
        symptoms = set()
        doc = self.nlp(text)
        text_lower = text.lower()

        symptom_patterns = [
            r'([a-z]+)\s+and\s+([a-z]+)\s+pain',
            r'([a-z]+)\s+pain',
            r'pain\s+in\s+(?:my|the)\s+([a-z]+)',
            r'my\s+([a-z]+)\s+(?:hurts?|aches?)',
        ]

        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptoms_found = []

                if 'and' in pattern:
                    body_part1 = match.group(1).strip()
                    body_part2 = match.group(2).strip()
                    symptoms_found.append(f"{body_part1} pain")
                    symptoms_found.append(f"{body_part2} pain")
                else:
                    body_part = match.group(1).strip()
                    symptoms_found.append(f"{body_part} pain")

                for symptom in symptoms_found:
                    if len(symptom) < 8:
                        continue

                    is_valid, conf = self.validator.is_valid_entity(symptom, 'symptom')
                    if is_valid and conf > 0.52:
                        symptom_clean = ' '.join(word.capitalize() for word in symptom.split())
                        symptoms.add(symptom_clean)

        specific_symptoms = [
            'headache', 'backache', 'nausea', 'dizziness', 'fatigue',
            'numbness', 'tingling', 'weakness', 'swelling', 'stiffness'
        ]

        for specific in specific_symptoms:
            if specific in text_lower:
                symptoms.add(specific.capitalize())

        return symptoms

    def extract_treatments_robust(self, text: str) -> Set[str]:
        treatments = set()
        text_lower = text.lower()

        sessions_pattern = r'(\d+)\s+sessions?\s+of\s+([a-z]+)'
        matches = re.finditer(sessions_pattern, text_lower)
        for match in matches:
            treatment = match.group(2).strip()
            if len(treatment) >= 5:
                is_valid, conf = self.validator.is_valid_entity(treatment, 'treatment')
                if is_valid and conf > 0.50:
                    treatments.add(treatment.capitalize())

        medication_keywords = ['painkiller', 'painkillers', 'medication', 'metformin', 'insulin']
        for med in medication_keywords:
            if med in text_lower:
                treatments.add(med.capitalize() if not med.endswith('s') else med.capitalize())

        therapy_pattern = r'([a-z]{8,})\s+(?:therapy|treatment)'
        matches = re.finditer(therapy_pattern, text_lower)
        for match in matches:
            therapy = match.group(1).strip()
            if len(therapy) >= 8:
                is_valid, conf = self.validator.is_valid_entity(therapy, 'treatment')
                if is_valid and conf > 0.50:
                    treatments.add(therapy.capitalize())

        drug_pattern = r'\b([A-Z][a-z]{4,}(?:in|ol|ate|ide|ine))\b'
        matches = re.finditer(drug_pattern, text)
        for match in matches:
            med = match.group(1).strip()
            if len(med) >= 6:
                is_valid, conf = self.validator.is_valid_entity(med, 'medication')
                if is_valid and conf > 0.60:
                    treatments.add(med)

        return treatments

    def extract_using_qa(self, text: str) -> Dict[str, str]:
        questions = {
            'diagnosis': "What medical condition, disease, or injury was diagnosed?",
            'current_status': "What is the patient's current condition or how are they feeling now?",
            'prognosis': "What is the expected recovery time or outcome?"
        }

        extracted = {}

        for key, question in questions.items():
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=text,
                    max_answer_len=100
                )

                if result['score'] > 0.30:
                    answer = result['answer'].strip()
                    answer = re.sub(r'^(a|an|the)\s+', '', answer, flags=re.IGNORECASE)
                    if len(answer) > 5:
                        extracted[key] = answer
            except:
                continue

        return extracted

    def generate_summary(self, transcript: str) -> MedicalSummary:
        print("Extracting medical entities...")

        qa_results = self.extract_using_qa(transcript)
        symptoms = self.extract_symptoms_robust(transcript)
        treatments = self.extract_treatments_robust(transcript)

        doc = self.nlp(transcript)
        patient_name = "Unknown"
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name = ent.text.replace('Mr.', '').replace('Ms.', '').replace('Mrs.', '').replace('Dr.', '').strip()
                if name and len(name) > 2 and name.lower() not in ['doctor', 'physician']:
                    patient_name = name
                    break

        diagnosis = qa_results.get('diagnosis', '')
        if not diagnosis or len(diagnosis) < 5:
            diagnosis_patterns = [
                r'(?:diagnosed with|said it was|looks like)\s+(?:a |an |you (?:may )?have (?:a |an )?)?([a-zA-Z\s]{5,40}?)(?:\.|,|;|\?)',
            ]
            for pattern in diagnosis_patterns:
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match:
                    diagnosis = match.group(1).strip().title()
                    break

        if not diagnosis or len(diagnosis) < 5:
            diagnosis = "To be determined"

        current_status = qa_results.get('current_status', 'Under observation')
        if len(current_status) > 120:
            current_status = current_status[:117] + "..."

        prognosis = qa_results.get('prognosis', 'Prognosis to be determined')
        if len(prognosis) > 150:
            prognosis = prognosis[:147] + "..."

        has_symptoms = len(symptoms) > 0
        has_diagnosis = diagnosis != "To be determined"
        has_treatment = len(treatments) > 0

        confidence = 0.50
        if has_diagnosis and has_symptoms:
            confidence += 0.25
        elif has_diagnosis or has_symptoms:
            confidence += 0.15
        if has_treatment:
            confidence += 0.10
        if len(qa_results) >= 2:
            confidence += 0.05

        confidence = max(0.50, min(0.90, confidence))

        return MedicalSummary(
            Patient_Name=patient_name,
            Symptoms=sorted(list(symptoms)) if symptoms else ["Not specified"],
            Diagnosis=diagnosis,
            Treatment=sorted(list(treatments)) if treatments else ["Not specified"],
            Current_Status=current_status,
            Prognosis=prognosis,
            Confidence_Score=float(confidence)
        )


class ClinicalSentimentAnalyzer:
    def __init__(self, use_gpu: bool = True):
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1

        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )

    def analyze(self, text: str) -> SentimentResult:
        if not text or len(text) < 10:
            return SentimentResult(Sentiment="Neutral", Intent="General conversation", Confidence=0.5)

        sentiment_result = self.zero_shot(
            text[:512],
            candidate_labels=["patient is anxious or worried", "patient is reassured and calm", "patient is neutral"],
            hypothesis_template="{}."
        )

        sentiment_map = {
            "patient is anxious or worried": "Anxious",
            "patient is reassured and calm": "Reassured",
            "patient is neutral": "Neutral"
        }

        sentiment = sentiment_map.get(sentiment_result['labels'][0], "Neutral")
        confidence = float(sentiment_result['scores'][0])

        intent_result = self.zero_shot(
            text[:512],
            candidate_labels=["seeking reassurance", "reporting symptoms", "seeking information", "expressing gratitude", "describing medical history"],
            hypothesis_template="The patient is {}."
        )

        intent = intent_result['labels'][0].title()

        return SentimentResult(
            Sentiment=sentiment,
            Intent=intent,
            Confidence=confidence
        )


class SOAPNoteGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

    def generate(self, transcript: str) -> SOAPNote:
        print("Generating SOAP note...")

        subjective_cc = self._qa_extract(transcript, "What is the patient's main complaint?")
        if not subjective_cc or len(subjective_cc) < 5:
            lines = transcript.split('\n')
            for line in lines:
                if 'patient:' in line.lower():
                    statement = line.split(':', 1)[1].strip()
                    if len(statement) > 10 and not statement.endswith('?'):
                        subjective_cc = statement[:150]
                        break

        subjective_hpi = self._extract_patient_statements(transcript)

        objective_exam = self._qa_extract(transcript, "What were the physical examination findings?")
        if not objective_exam or len(objective_exam) < 10:
            objective_exam = "Not documented"

        objective_obs = self._qa_extract(transcript, "What observations were made about the patient?")
        if not objective_obs or len(objective_obs) < 5:
            objective_obs = "Patient appears stable"

        assessment_dx = self._qa_extract(transcript, "What was the diagnosis?")
        if not assessment_dx or len(assessment_dx) < 5:
            patterns = [
                r'(?:diagnosed with|said it was)\s+(?:a |an )?([a-zA-Z\s]{5,40}?)(?:\.|,)'
            ]
            for pattern in patterns:
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match:
                    assessment_dx = match.group(1).strip().title()
                    break
        if not assessment_dx or len(assessment_dx) < 5:
            assessment_dx = "To be determined"

        assessment_severity = self._determine_severity(transcript)

        plan_tx = self._extract_treatment_plan(transcript)
        plan_followup = self._extract_followup(transcript)

        return SOAPNote(
            Subjective={
                "Chief_Complaint": subjective_cc if subjective_cc else "Not specified",
                "History_of_Present_Illness": subjective_hpi[:500] if subjective_hpi else "Not documented"
            },
            Objective={
                "Physical_Exam": objective_exam,
                "Observations": objective_obs
            },
            Assessment={
                "Diagnosis": assessment_dx,
                "Severity": assessment_severity
            },
            Plan={
                "Treatment_Plan": plan_tx if plan_tx else "Continue current management",
                "Follow-Up": plan_followup if plan_followup else "Return as needed (PRN)"
            }
        )

    def _qa_extract(self, text: str, question: str) -> str:
        try:
            result = self.qa(question=question, context=text, max_answer_len=100)
            if result['score'] > 0.25:
                return result['answer'].strip()
        except:
            pass
        return ""

    def _extract_patient_statements(self, transcript: str) -> str:
        lines = transcript.split('\n')
        statements = []

        for line in lines:
            if 'patient:' in line.lower():
                content = line.split(':', 1)[1].strip()
                if content and len(content) > 10:
                    statements.append(content)

        return " ".join(statements[:5])

    def _extract_treatment_plan(self, transcript: str) -> str:
        sentences = [sent.text for sent in self.nlp(transcript).sents]
        treatment_sentences = []

        treatment_keywords = ['prescribe', 'recommend', 'start', 'continue', 'medication', 'therapy', 'treatment', 'sessions']
        patient_keywords = ['had to', 'went through', 'have to', 'will have', 'need to']

        for sent in sentences:
            sent_lower = sent.lower()

            if 'physician:' in sent_lower or 'doctor:' in sent_lower:
                if any(kw in sent_lower for kw in treatment_keywords):
                    if not sent.strip().endswith('?'):
                        treatment_sentences.append(sent.strip())
            elif any(kw in sent_lower for kw in patient_keywords):
                if any(tx in sent_lower for tx in ['physiotherapy', 'therapy', 'medication', 'sessions']):
                    continue

        if treatment_sentences:
            return ' '.join(treatment_sentences[:2])[:300]

        return ""

    def _extract_followup(self, transcript: str) -> str:
        followup_patterns = [
            r'(?:schedule|come back|return)\s+(?:for\s+)?(?:a\s+)?follow[- ]?up\s+([^.]{5,80})',
            r'follow[- ]?up\s+(?:in|after|next)\s+([^.]{5,50})',
        ]

        for pattern in followup_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                followup = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
                followup = re.sub(r'\s+', ' ', followup)
                if 10 < len(followup) < 100:
                    return followup.capitalize()

        if 'follow up' in transcript.lower() or 'follow-up' in transcript.lower():
            return "Follow-up appointment to be scheduled"

        return ""

    def _determine_severity(self, text: str) -> str:
        text_lower = text.lower()

        if any(word in text_lower for word in ['severe', 'critical', 'acute']):
            return "Severe"
        elif any(word in text_lower for word in ['improving', 'better', 'recovery']):
            return "Mild, improving"
        elif any(word in text_lower for word in ['mild', 'minor']):
            return "Mild"
        else:
            return "Moderate"


class ClinicalTranscriptionPipeline:
    def __init__(self, use_gpu: bool = True):

        print("INITIALIZING CLINICAL PIPELINE")


        self.extractor = EnhancedClinicalExtractor(use_gpu=use_gpu)
        self.sentiment_analyzer = ClinicalSentimentAnalyzer(use_gpu=use_gpu)
        self.soap_generator = SOAPNoteGenerator()

    def save_results(self, results, filename="results.json"):
        # Get the current directory where the script is running
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, filename)

        # Save JSON results
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved successfully at: {save_path}")

    def process_transcript(self, transcript: str, patient_text: str = None) -> Dict:
        print("PROCESSING TRANSCRIPT\n")

        summary = self.extractor.generate_summary(transcript)

        print("Analyzing sentiment...")
        if not patient_text:
            lines = [l.split(':', 1)[1].strip() for l in transcript.split('\n')
                    if 'patient:' in l.lower() and ':' in l]
            patient_text = " ".join(lines[:3])

        sentiment = self.sentiment_analyzer.analyze(patient_text)

        soap = self.soap_generator.generate(transcript)

        print("\n✓ PROCESSING COMPLETE\n")

        return {
            "medical_summary": asdict(summary),
            "sentiment_analysis": asdict(sentiment),
            "soap_note": asdict(soap)
        }

    def save_results(self, results: Dict, filename: str = "clinical_results.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {filename}")


if __name__ == "__main__":
    SAMPLE = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That's a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
"""



    pipeline = ClinicalTranscriptionPipeline(use_gpu=torch.cuda.is_available())
    results = pipeline.process_transcript(SAMPLE)



    print(" MEDICAL SUMMARY:")
    print(json.dumps(results['medical_summary'], indent=2))

    print("\n SENTIMENT ANALYSIS:")
    print(json.dumps(results['sentiment_analysis'], indent=2))

    print("\n SOAP NOTE:")
    print(json.dumps(results['soap_note'], indent=2))


    print("✓ COMPLETE")


    pipeline.save_results(results)