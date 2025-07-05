"""
Knowledge Graph Builder Agent - Build knowledge graph từ extracted features
Entity Extraction, Relation Building, Graph Connection, Schema Design  
Phần của Phase 1: Preprocessing
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import time

from ..base_agent import PreprocessingAgent, AgentResponse

# Structured output models
class Entity(BaseModel):
    """Entity trong knowledge graph"""
    entity_id: str = Field(..., description="Unique entity ID")
    entity_type: str = Field(..., description="Entity type: person, object, location, concept, etc.")
    name: str = Field(..., description="Entity name")
    description: str = Field(..., description="Entity description")
    
    # Properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    
    # Source information
    sources: List[str] = Field(default_factory=list, description="Source video/frame IDs")
    timestamps: List[float] = Field(default_factory=list, description="Timestamps where entity appears")
    confidence: float = Field(..., description="Entity extraction confidence")
    
    # Embeddings
    text_embedding: List[float] = Field(default_factory=list, description="Text embedding")
    visual_embedding: List[float] = Field(default_factory=list, description="Visual embedding if applicable")

class Relation(BaseModel):
    """Relationship giữa entities"""
    relation_id: str = Field(..., description="Unique relation ID")
    relation_type: str = Field(..., description="Relation type: contains, interacts_with, located_in, etc.")
    
    # Connected entities
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    
    # Relation properties
    description: str = Field(..., description="Relation description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relation properties")
    confidence: float = Field(..., description="Relation confidence")
    
    # Temporal info
    start_time: Optional[float] = Field(None, description="When relation starts")
    end_time: Optional[float] = Field(None, description="When relation ends")
    duration: Optional[float] = Field(None, description="Relation duration")
    
    # Evidence
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this relation")
    sources: List[str] = Field(default_factory=list, description="Source references")

class GraphSchema(BaseModel):
    """Schema definition cho knowledge graph"""
    schema_version: str = Field(..., description="Schema version")
    
    # Entity types
    entity_types: List[Dict[str, Any]] = Field(default_factory=list, description="Supported entity types")
    
    # Relation types  
    relation_types: List[Dict[str, Any]] = Field(default_factory=list, description="Supported relation types")
    
    # Constraints
    constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Graph constraints")
    
    # Indexes
    indexes: List[str] = Field(default_factory=list, description="Recommended indexes")

class KnowledgeGraph(BaseModel):
    """Kết quả knowledge graph construction"""
    graph_id: str = Field(..., description="Graph ID")
    video_id: str = Field(..., description="Source video ID")
    creation_timestamp: str = Field(..., description="Graph creation time")
    
    # Graph content
    entities: List[Entity] = Field(default_factory=list, description="All entities")
    relations: List[Relation] = Field(default_factory=list, description="All relations")
    schema: GraphSchema = Field(..., description="Graph schema")
    
    # Statistics
    stats: Dict[str, Any] = Field(default_factory=dict, description="Graph statistics")
    
    # Processing info
    processing_time: float = Field(..., description="Construction time")
    extraction_methods: List[str] = Field(default_factory=list, description="Methods used")
    status: str = Field(..., description="Construction status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class GraphBuildingConfig(BaseModel):
    """Configuration cho knowledge graph building"""
    # Entity extraction
    extract_persons: bool = Field(default=True, description="Extract person entities")
    extract_objects: bool = Field(default=True, description="Extract object entities")
    extract_locations: bool = Field(default=True, description="Extract location entities")
    extract_concepts: bool = Field(default=True, description="Extract concept entities")
    extract_actions: bool = Field(default=True, description="Extract action entities")
    
    # Relation extraction
    extract_spatial_relations: bool = Field(default=True, description="Extract spatial relations")
    extract_temporal_relations: bool = Field(default=True, description="Extract temporal relations")
    extract_causal_relations: bool = Field(default=True, description="Extract causal relations")
    extract_interaction_relations: bool = Field(default=True, description="Extract interaction relations")
    
    # Thresholds
    min_entity_confidence: float = Field(default=0.7, description="Minimum entity confidence")
    min_relation_confidence: float = Field(default=0.6, description="Minimum relation confidence")
    max_entities_per_frame: int = Field(default=20, description="Max entities per frame")
    
    # Graph optimization
    merge_similar_entities: bool = Field(default=True, description="Merge similar entities")
    similarity_threshold: float = Field(default=0.85, description="Entity similarity threshold")
    remove_weak_relations: bool = Field(default=True, description="Remove low-confidence relations")

class KnowledgeGraphAgent(PreprocessingAgent):
    """
    Agent chuyên build knowledge graph từ extracted features
    Thực hiện entity extraction, relation building, schema design
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="KnowledgeGraphBuilder",
            model_type="gpt-4o",  # Cần model mạnh cho knowledge reasoning
            **kwargs
        )
        
        self.set_instructions([
            "You are a knowledge graph construction specialist.",
            "Build comprehensive knowledge graphs from multi-modal video features.",
            "Tasks include:",
            "- Extract entities (persons, objects, locations, concepts, actions)",
            "- Identify relationships between entities",
            "- Design optimal graph schemas",
            "- Handle temporal and spatial relationships",
            "- Merge and deduplicate similar entities",
            "- Optimize graph structure for search and reasoning",
            "Focus on accuracy, completeness, and semantic richness.",
            "Always provide confidence scores and evidence for extractions."
        ])
        
        self.agent.response_model = KnowledgeGraph
        
    def process(self, features_data: Dict[str, Any], config: Optional[GraphBuildingConfig] = None, **kwargs) -> AgentResponse:
        """
        Build knowledge graph từ extracted features
        
        Args:
            features_data: Multi-modal features từ FeatureExtractorAgent
            config: Graph building configuration  
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Use default config if not provided
            if config is None:
                config = GraphBuildingConfig()
            
            # Create graph building prompt
            prompt = self._create_graph_prompt(features_data, config)
            
            # Run agent với structured output
            response = self.agent.run(prompt)
            
            # Validate response
            if not isinstance(response, KnowledgeGraph):
                result = self._parse_graph_response(response.content, features_data)
            else:
                result = response
            
            # Post-process graph nếu cần
            if config.merge_similar_entities:
                result = self._merge_similar_entities(result, config.similarity_threshold)
                
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="knowledge_graph_building",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "entity_count": len(result.entities),
                    "relation_count": len(result.relations),
                    "config": config.dict(),
                    "graph_density": len(result.relations) / max(len(result.entities), 1)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Knowledge graph building failed: {str(e)}")
            
            return self._create_response(
                task_type="knowledge_graph_building",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_graph_prompt(self, features_data: Dict[str, Any], config: GraphBuildingConfig) -> str:
        """Tạo prompt cho knowledge graph construction"""
        
        # Extract feature summaries
        visual_features = features_data.get("visual_features", [])
        audio_features = features_data.get("audio_features", [])
        video_id = features_data.get("video_id", "unknown")
        
        prompt = f"""
        Build a comprehensive knowledge graph from multi-modal features for video: {video_id}
        
        INPUT DATA SUMMARY:
        - Visual Features: {len(visual_features)} keyframes with CLIP/BLIP features and OCR
        - Audio Features: {len(audio_features)} segments with speech transcripts and audio analysis
        
        CONFIGURATION:
        - Extract Persons: {config.extract_persons}
        - Extract Objects: {config.extract_objects}  
        - Extract Locations: {config.extract_locations}
        - Extract Concepts: {config.extract_concepts}
        - Extract Actions: {config.extract_actions}
        - Extract Spatial Relations: {config.extract_spatial_relations}
        - Extract Temporal Relations: {config.extract_temporal_relations}
        - Minimum Entity Confidence: {config.min_entity_confidence}
        - Minimum Relation Confidence: {config.min_relation_confidence}
        
        FEATURE DATA:
        """
        
        # Add visual feature details
        if visual_features:
            prompt += "\nVISUAL FEATURES:\n"
            for i, vf in enumerate(visual_features[:5]):  # Limit để không quá dài
                prompt += f"Frame {i+1} (t={vf.get('timestamp', 0)}s):\n"
                prompt += f"  - BLIP Caption: {vf.get('blip_caption', 'N/A')}\n"
                prompt += f"  - CLIP Description: {vf.get('clip_description', 'N/A')}\n"
                prompt += f"  - Objects: {vf.get('blip_objects', [])}\n"
                prompt += f"  - OCR Text: {vf.get('ocr_text', 'N/A')}\n"
                prompt += f"  - Scene Type: {vf.get('scene_type', 'N/A')}\n"
        
        # Add audio feature details  
        if audio_features:
            prompt += "\nAUDIO FEATURES:\n"
            for i, af in enumerate(audio_features[:5]):  # Limit để không quá dài
                prompt += f"Segment {i+1} (t={af.get('start_time', 0)}-{af.get('end_time', 0)}s):\n"
                prompt += f"  - Transcript: {af.get('transcript', 'N/A')}\n"
                prompt += f"  - Language: {af.get('language', 'N/A')}\n"
                prompt += f"  - Audio Type: {af.get('audio_type', 'N/A')}\n"
                prompt += f"  - Emotion: {af.get('emotion', 'N/A')}\n"
                prompt += f"  - Speaker Count: {af.get('speaker_count', 0)}\n"
        
        prompt += f"""
        
        GRAPH CONSTRUCTION TASKS:
        
        1. ENTITY EXTRACTION:
           - Extract all relevant entities from visual and audio content
           - Include entity types: persons, objects, locations, concepts, actions
           - Assign confidence scores >= {config.min_entity_confidence}
           - Generate unique IDs and comprehensive descriptions
           - Map entities to their source timestamps and frames
           - Extract entity properties and aliases
        
        2. RELATION EXTRACTION:
           - Identify relationships between extracted entities
           - Include spatial, temporal, causal, and interaction relations
           - Assign confidence scores >= {config.min_relation_confidence}
           - Map relations to temporal segments where they occur
           - Provide evidence from source data
        
        3. SCHEMA DESIGN:
           - Design optimal schema for this video domain
           - Define entity and relation type hierarchies
           - Specify constraints and validation rules
           - Recommend indexes for efficient querying
        
        4. GRAPH OPTIMIZATION:
           - Ensure consistency across temporal segments
           - Connect related entities across different modalities
           - Generate comprehensive statistics
           - Validate graph structure and completeness
        
        OUTPUT REQUIREMENTS:
        - Return complete KnowledgeGraph object
        - Include all entities với detailed properties
        - Include all relations với evidence
        - Provide comprehensive schema definition
        - Generate detailed statistics and metadata
        """
        
        return prompt
    
    def _parse_graph_response(self, response_content: str, features_data: Dict) -> KnowledgeGraph:
        """Fallback parsing nếu structured output fails"""
        return KnowledgeGraph(
            graph_id=f"graph_{int(time.time())}",
            video_id=features_data.get("video_id", "unknown"),
            creation_timestamp=str(time.time()),
            entities=[],
            relations=[],
            schema=GraphSchema(
                schema_version="1.0",
                entity_types=[],
                relation_types=[],
                constraints=[],
                indexes=[]
            ),
            stats={},
            processing_time=0,
            extraction_methods=["fallback_parser"],
            status="parsed_fallback"
        )
    
    def _merge_similar_entities(self, graph: KnowledgeGraph, threshold: float) -> KnowledgeGraph:
        """Merge similar entities để reduce duplication"""
        # Simplified merging logic - trong thực tế sẽ complex hơn
        merged_entities = []
        entity_map = {}  # old_id -> new_id mapping
        
        for entity in graph.entities:
            # Find similar entities based on name and type
            similar_found = False
            for merged in merged_entities:
                if (entity.entity_type == merged.entity_type and 
                    entity.name.lower() == merged.name.lower()):
                    # Merge into existing entity
                    merged.sources.extend(entity.sources)
                    merged.timestamps.extend(entity.timestamps)
                    merged.aliases.extend(entity.aliases)
                    entity_map[entity.entity_id] = merged.entity_id
                    similar_found = True
                    break
            
            if not similar_found:
                merged_entities.append(entity)
                entity_map[entity.entity_id] = entity.entity_id
        
        # Update relations với new entity IDs
        updated_relations = []
        for relation in graph.relations:
            if (relation.source_entity_id in entity_map and 
                relation.target_entity_id in entity_map):
                relation.source_entity_id = entity_map[relation.source_entity_id]
                relation.target_entity_id = entity_map[relation.target_entity_id]
                updated_relations.append(relation)
        
        graph.entities = merged_entities
        graph.relations = updated_relations
        graph.stats["entities_merged"] = len(graph.entities) - len(merged_entities)
        
        return graph
    
    def build_from_visual_only(self, visual_features: List[Dict], **kwargs) -> AgentResponse:
        """Build graph chỉ từ visual features"""
        features_data = {
            "video_id": f"visual_only_{int(time.time())}",
            "visual_features": visual_features,
            "audio_features": []
        }
        
        config = GraphBuildingConfig(
            extract_persons=True,
            extract_objects=True,
            extract_locations=True,
            extract_concepts=False,
            extract_actions=False
        )
        
        return self.process(features_data, config, **kwargs)
    
    def build_from_audio_only(self, audio_features: List[Dict], **kwargs) -> AgentResponse:
        """Build graph chỉ từ audio features""" 
        features_data = {
            "video_id": f"audio_only_{int(time.time())}",
            "visual_features": [],
            "audio_features": audio_features
        }
        
        config = GraphBuildingConfig(
            extract_persons=True,
            extract_objects=False,
            extract_locations=False,
            extract_concepts=True,
            extract_actions=True
        )
        
        return self.process(features_data, config, **kwargs)
    
    def analyze_graph_quality(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze chất lượng của knowledge graph"""
        entities = graph_data.get("entities", [])
        relations = graph_data.get("relations", [])
        
        if not entities:
            return {"quality_score": 0, "issues": ["No entities found"]}
        
        # Calculate quality metrics
        entity_count = len(entities)
        relation_count = len(relations)
        avg_entity_confidence = sum(e.get("confidence", 0) for e in entities) / entity_count
        avg_relation_confidence = sum(r.get("confidence", 0) for r in relations) / max(relation_count, 1)
        
        # Graph connectivity
        connected_entities = set()
        for relation in relations:
            connected_entities.add(relation.get("source_entity_id"))
            connected_entities.add(relation.get("target_entity_id"))
        
        connectivity_ratio = len(connected_entities) / entity_count
        graph_density = relation_count / max(entity_count * (entity_count - 1) / 2, 1)
        
        # Calculate overall quality score
        quality_score = (
            avg_entity_confidence * 0.3 +
            avg_relation_confidence * 0.3 +
            connectivity_ratio * 0.2 +
            min(graph_density * 10, 1.0) * 0.2
        )
        
        return {
            "quality_score": round(quality_score, 3),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "avg_entity_confidence": round(avg_entity_confidence, 3),
            "avg_relation_confidence": round(avg_relation_confidence, 3),
            "connectivity_ratio": round(connectivity_ratio, 3),
            "graph_density": round(graph_density, 4),
            "recommendations": self._generate_quality_recommendations(
                avg_entity_confidence, avg_relation_confidence, connectivity_ratio
            )
        }
    
    def _generate_quality_recommendations(self, entity_conf: float, relation_conf: float, connectivity: float) -> List[str]:
        """Generate recommendations để improve graph quality"""
        recommendations = []
        
        if entity_conf < 0.7:
            recommendations.append("Increase minimum entity confidence threshold")
        
        if relation_conf < 0.6:
            recommendations.append("Improve relation extraction accuracy")
            
        if connectivity < 0.5:
            recommendations.append("Extract more connecting relationships between entities")
            
        if not recommendations:
            recommendations.append("Graph quality is good, consider adding more entity types")
            
        return recommendations