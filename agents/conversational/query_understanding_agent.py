import spacy
from typing import Dict, List, Any
from agents.base_agent import BaseAgent
# from agno.models.openai import OpenAIChat  # Uncomment if available
# from tools.conversation.context_tools import extract_intent, extract_entities  # Uncomment if available

class QueryUnderstandingAgent(BaseAgent):
    def __init__(self, model: Any, name: str = "QueryUnderstandingAgent", role: str = "query_understanding", config: Dict[str, Any] = None):
        super().__init__(model, name, role, config)
        self.nlp = spacy.blank("vi")  # Use blank Vietnamese model, replace with actual model if available

    async def understand_query(self, query: str, context: Dict) -> Dict[str, Any]:
        try:
            intent = self.extract_search_intent(query)
            entities = self.extract_entities(query)
            expanded_query = self.expand_query(query, entities)
            search_filters = self.build_search_filters(entities, context)
            return {
                "intent": intent,
                "entities": entities,
                "expanded_query": expanded_query,
                "search_filters": search_filters
            }
        except Exception as e:
            return self.handle_error(e, {"query": query, "context": context})

    def extract_search_intent(self, query: str) -> str:
        # Dummy intent classification, replace with actual logic or model
        if "tìm" in query.lower():
            return "video_search"
        elif "giải thích" in query.lower():
            return "explanation_request"
        else:
            return "general_query"

    def extract_entities(self, query: str) -> List[str]:
        # Dummy entity extraction, replace with spaCy or custom NER
        doc = self.nlp(query)
        # For demo, just split by space and filter stopwords
        entities = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return entities

    def expand_query(self, original_query: str, entities: List[str]) -> List[str]:
        # Dummy expansion, replace with semantic similarity or synonyms
        expanded = [original_query] + entities
        return expanded

    def build_search_filters(self, entities: List[str], context: Dict) -> Dict[str, Any]:
        # Dummy filter builder, can be extended
        return {"entities": entities, "user_id": context.get("user_id")}

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý input_data dạng {'query': ..., 'context': ...} và trả về kết quả như understand_query
        """
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        return await self.understand_query(query, context)
