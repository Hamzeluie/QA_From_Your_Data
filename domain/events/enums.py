from enum import Enum

class ProcessSteps(str, Enum):
    SAVE_CANONICAL_POSTGRES = "save_canonical_posgres"
    SAVE_CANONICAL_NEO4J = "save_canonical_neo4j"
    SAVE_CANONICAL_QDRANT = "save_canonical_qdrant"
    SAVE_MENTION_POSTGRES = "save_mention_postgres"
    SAVE_CHUNKS_QDRANT = "save_chunks_gdrant"
    UPDATE_CHUNK_CANONICALS_QDRANT = "update_chunk_canonicals_qdrant"
    SAVE_CHUNKS_ELASTICSEARCH = "save_chunks_elasticsearch"   
    SAVE_RELATION_POSTGRES = "save_relation_postgres"
    SAVE_RELATION_NEO4J = "save_relation_neo4j"
    SAVE_RESOLVED_MENTION_AS_CANONICAL_PAYLOAD = "save_resolved_mention_as_canonical"
    DELETE_RESOLVED_MENTION_POSTGRES = "delete_resolved_mention_postgres"
    DELETE_MENTION_PAYLOAD = "delete_mention_payload"
    

class EventType(str, Enum):
    UPLOAD_RAW_TEXT = "upload_raw_text"
    RESOLVED_ENTITY = "resolved_entity"
    HUMAN_RESOLVED_MENTION = "human_resolved_mention"
    RELATION_EXTRACTION = "realtion_extraction"
    DELETE_CANONICAL = "delete_canonical"
    DELETE_MENTION = "delete_mention"
    UPDATE_CANONICAL = "update_canonical"
    UPDATE_CHUNK_METADATA = "update_chunk_metadata"
    UPDATE_RELATION = "update_relation"
    
class DocumentStatus(str, Enum):
    FAILD = "faild"
    UPLOADED = "upload"
    ENTITIES_RESOLVING = "entities_resolving"
    ENTITIES_RESOLVED = "entities_resolved"
    NEEDS_REVIEW = "needs_review"
    EXTRACTING_RELATIONS = "extracting_relations"
    RELATION_EXTRACTED = "relation_extracted"
    
