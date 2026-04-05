import google.generativeai as genai
import json
from typing import Dict, List
from config import Config


class LLMService:
    def __init__(self):
        self.has_api_key = Config.GOOGLE_GEMINI_API_KEY is not None
        if self.has_api_key:
            genai.configure(api_key=Config.GOOGLE_GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.LLM_MODEL)
            Config.log_event("LLM_INITIALIZED", "Google Gemini API initialized")
        else:
            self.model = None
            Config.log_event(
                "LLM_FALLBACK",
                "No Google Gemini API key; using template-based summaries"
            )

    # -------------------------------------------------
    # SINGLE TOPIC LABEL (BACKWARD COMPATIBLE)
    # -------------------------------------------------
    def label_topic(self, keywords: List[str]) -> str:
        if not keywords:
            return "Unknown Topic"

        if not self.has_api_key:
            return " ".join(keywords[:2]).title()

        try:
            keywords_str = ", ".join(keywords)
            prompt = f"""
Task: Categorize this research topic into a standard academic field name (2–4 words max).
Keywords: {keywords_str}
Label:
"""
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 10,
                    "temperature": 0.1
                }
            )
            return response.text.strip().replace("\n", "").replace("Label:", "").strip()

        except Exception as e:
            Config.log_event("LLM_ERROR", f"Topic labeling failed: {e}")
            return " ".join(keywords[:2]).title()

    # -------------------------------------------------
    # 🔥 BATCH TOPIC LABELING (ONE GEMINI CALL)
    # -------------------------------------------------
    def label_topics_batch(self, topics: List[Dict]) -> Dict[int, str]:
        """
        topics = [
            {"id": 0, "keywords": ["llm", "transformer", "training"]},
            {"id": 1, "keywords": ["benchmark", "evaluation", "performance"]}
        ]
        returns -> {0: "Large Language Models", 1: "Model Evaluation"}
        """

        if not topics:
            return {}

        # Fallback (no API key)
        if not self.has_api_key:
            return {
                int(t["id"]): " / ".join(t["keywords"][:2]).title()
                for t in topics
            }

        # Build compact prompt
        topic_lines = []
        for t in topics:
            kws = ", ".join(t["keywords"][:6])
            topic_lines.append(f"Topic {t['id']}: {kws}")

        prompt = f"""
You are an expert research analyst.

Given the following topic keywords extracted from scientific papers,
assign a SHORT, HUMAN-READABLE topic label (2–5 words max) for each topic.

Return ONLY valid JSON in this exact format:
{{ "topic_id": "label" }}

Topics:
{chr(10).join(topic_lines)}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 200,
                    "temperature": 0.1
                }
            )

            text = response.text.strip()

            # Extract JSON safely
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]

            labels = json.loads(json_str)

            return {int(k): str(v) for k, v in labels.items()}

        except Exception as e:
            Config.log_event("LLM_ERROR", f"Batch topic labeling failed: {e}")

            # Safe fallback
            return {
                int(t["id"]): " / ".join(t["keywords"][:2]).title()
                for t in topics
            }

    # -------------------------------------------------
    # EXISTING SUMMARY METHODS (UNCHANGED)
    # -------------------------------------------------
    def summarize_topic_evolution(self, evolution_data: Dict) -> str:
        if not self.has_api_key:
            return self._template_topic_evolution_summary(evolution_data)
        try:
            prompt = (
                "Based on the following research topic evolution data, "
                "provide a concise (2–3 sentences) summary.\n"
                f"Data: {evolution_data}"
            )
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": Config.LLM_MAX_TOKENS,
                    "temperature": Config.LLM_TEMPERATURE
                }
            )
            return response.text
        except Exception:
            return self._template_topic_evolution_summary(evolution_data)

    def summarize_collaboration_influence(self, collaboration_data: Dict) -> str:
        if not self.has_api_key:
            return self._template_collaboration_summary(collaboration_data)
        try:
            prompt = f"Summarize collaboration influence.\nData: {collaboration_data}"
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": Config.LLM_MAX_TOKENS,
                    "temperature": Config.LLM_TEMPERATURE
                }
            )
            return response.text
        except Exception:
            return self._template_collaboration_summary(collaboration_data)

    def generate_prediction_narrative(self, predictions: Dict) -> str:
        if not self.has_api_key:
            return self._template_prediction_narrative(predictions)
        try:
            prompt = f"Predict future research directions.\nData: {predictions}"
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": Config.LLM_MAX_TOKENS,
                    "temperature": Config.LLM_TEMPERATURE
                }
            )
            return response.text
        except Exception:
            return self._template_prediction_narrative(predictions)

    # -------------------------------------------------
    # FALLBACK TEMPLATES
    # -------------------------------------------------
    @staticmethod
    def _template_topic_evolution_summary(evolution_data):
        return "The researcher's focus has remained relatively consistent over time."

    @staticmethod
    def _template_collaboration_summary(collaboration_data):
        return "Collaboration patterns indicate limited long-term partnerships."

    @staticmethod
    def _template_prediction_narrative(predictions):
        return "Insufficient data to confidently predict future research directions."


def get_llm_service() -> LLMService:
    return LLMService()
