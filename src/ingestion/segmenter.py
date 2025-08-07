import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class DocumentSegment:
    content: str
    segment_type: str  # 'clause', 'table', 'list', 'paragraph'
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]


class DocumentSegmenter:
    def __init__(self, config_path: str = "ingestion_config.yaml"):
        """Initialize document segmenter with configuration."""
        self.config = self._load_config(config_path)
        self.segmentation_config = self.config.get("ingestion", {}).get("segmentation", {})
        
        self.min_chunk_size = self.segmentation_config.get("min_chunk_size", 100)
        self.max_chunk_size = self.segmentation_config.get("max_chunk_size", 1000)
        self.overlap = self.segmentation_config.get("overlap", 50)
    
    def _load_config(self, config_path: str) -> Dict:
        config_path = os.path.abspath(config_path)
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _extract_tables(self, content: str) -> List[DocumentSegment]:
        """Extract table structures from content."""
        # Simple regex for Markdown tables: header|---|row
        table_pattern = re.compile(
            r'(?:^|\n)((?:\|[^\n]+\|\n)+\|(?:\s*-+\s*\|)+\n(?:\|[^\n]+\|\n?)+)', re.MULTILINE)
        matches = list(table_pattern.finditer(content))
        segments = []
        for m in matches:
            segments.append(DocumentSegment(
                content=m.group(1),
                segment_type='table',
                start_pos=m.start(1),
                end_pos=m.end(1),
                metadata={}
            ))
        return segments

    def segment_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentSegment]:
        """
        Segment document content into logical chunks.
        
        Args:
            content: Document content to segment
            metadata: Additional metadata about the document
            
        Returns:
            List of document segments
        """
        segments = []
        
        # First, identify and extract tables
        table_segments = self._extract_tables(content)
        segments.extend(table_segments)
        
        # Remove table content from main text for further processing
        content_without_tables = self._remove_table_content(content, table_segments)
        
        # Extract lists
        list_segments = self._extract_lists(content_without_tables)
        segments.extend(list_segments)
        
        # Remove list content from main text
        content_without_lists = self._remove_list_content(content_without_tables, list_segments)
        
        # Extract clauses and paragraphs
        clause_segments = self._extract_clauses(content_without_lists)
        segments.extend(clause_segments)
        
        # Sort segments by position
        segments.sort(key=lambda x: x.start_pos)
        
        return segments
    
    def _extract_lists(self, content: str) -> List[DocumentSegment]:
        """Extract list structures from content."""
        list_segments = []
        # List patterns: bullet, numbered, lettered
        list_patterns = [
            r'((?:^\s*[-*+]\s+.*(?:\n\s*[-*+]\s+.*)+))',  # Bullet lists
            r'((?:^\s*\d+\.\s+.*(?:\n\s*\d+\.\s+.*)+))',  # Numbered lists
            r'((?:^\s*[a-zA-Z]\.\s+.*(?:\n\s*[a-zA-Z]\.\s+.*)+))',  # Letter lists
        ]
        for pattern in list_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                list_content = match.group(1)
                if len(list_content.strip()) > 10:  # Lower threshold for test
                    segment = DocumentSegment(
                        content=list_content.strip(),
                        segment_type='list',
                        start_pos=match.start(),
                        end_pos=match.end(),
                        metadata={
                            'list_type': 'detected',
                            'num_items': len(re.findall(r'^\s*[-*+\d\.a-zA-Z]\.\s+', list_content, re.MULTILINE)),
                            'raw_match': match.group(0)
                        }
                    )
                    list_segments.append(segment)
        return list_segments
    
    def _extract_clauses(self, content: str) -> List[DocumentSegment]:
        """Extract clauses and paragraphs from content."""
        clause_segments = []
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_pos = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                current_pos += len(paragraph) + 2  # +2 for double newline
                continue
            
            # Check if paragraph is a clause (contains legal/contract language)
            is_clause = self._is_clause(paragraph)
            
            # Split long paragraphs into smaller chunks
            if len(paragraph) > self.max_chunk_size:
                chunks = self._split_long_paragraph(paragraph)
                for i, chunk in enumerate(chunks):
                    segment = DocumentSegment(
                        content=chunk,
                        segment_type='clause' if is_clause else 'paragraph',
                        start_pos=current_pos,
                        end_pos=current_pos + len(chunk),
                        metadata={
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'is_clause': is_clause
                        }
                    )
                    clause_segments.append(segment)
                    current_pos += len(chunk) + 1  # +1 for space
            else:
                segment = DocumentSegment(
                    content=paragraph,
                    segment_type='clause' if is_clause else 'paragraph',
                    start_pos=current_pos,
                    end_pos=current_pos + len(paragraph),
                    metadata={
                        'is_clause': is_clause,
                        'length': len(paragraph)
                    }
                )
                clause_segments.append(segment)
            
            current_pos += len(paragraph) + 2  # +2 for double newline
        
        return clause_segments
    
    def _is_clause(self, text: str) -> bool:
        """Determine if text contains legal/contract clause language."""
        clause_indicators = [
            r'\b(?:whereas|provided that|subject to|in the event|if|unless)\b',
            r'\b(?:party|parties|agreement|contract|terms|conditions)\b',
            r'\b(?:shall|will|must|may|can|cannot)\b',
            r'\b(?:liability|damages|compensation|payment|obligation)\b',
            r'\b(?:terminate|termination|breach|default|enforcement)\b',
            r'[A-Z][A-Z\s]+:',  # All caps headers
            r'\d+\.\s+[A-Z]',  # Numbered clauses
        ]
        
        text_lower = text.lower()
        clause_score = 0
        
        for pattern in clause_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                clause_score += 1
        
        return clause_score >= 2  # At least 2 indicators
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split long paragraphs into smaller chunks with overlap."""
        chunks = []
        
        if len(paragraph) <= self.max_chunk_size:
            return [paragraph]
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap:] if self.overlap > 0 else ""
                    current_chunk = overlap_text + sentence + " "
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) <= self.max_chunk_size:
                            current_chunk += word + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = word + " "
                            else:
                                # Single word is too long, truncate
                                chunks.append(word[:self.max_chunk_size])
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _remove_table_content(self, content: str, table_segments: List[DocumentSegment]) -> str:
        """Remove table content from main text."""
        # Sort segments by position in reverse order to avoid index shifting
        sorted_segments = sorted(table_segments, key=lambda x: x.start_pos, reverse=True)
        
        for segment in sorted_segments:
            content = content[:segment.start_pos] + content[segment.end_pos:]
        
        return content
    
    def _remove_list_content(self, content: str, list_segments: List[DocumentSegment]) -> str:
        """Remove list content from main text."""
        # Sort segments by position in reverse order to avoid index shifting
        sorted_segments = sorted(list_segments, key=lambda x: x.start_pos, reverse=True)
        
        for segment in sorted_segments:
            content = content[:segment.start_pos] + content[segment.end_pos:]
        
        return content
    
    def get_segment_statistics(self, segments: List[DocumentSegment]) -> Dict[str, Any]:
        """Get statistics about document segments."""
        stats = {
            'total_segments': len(segments),
            'segment_types': {},
            'total_content_length': 0,
            'avg_segment_length': 0
        }
        
        for segment in segments:
            # Count segment types
            segment_type = segment.segment_type
            if segment_type not in stats['segment_types']:
                stats['segment_types'][segment_type] = 0
            stats['segment_types'][segment_type] += 1
            
            # Calculate lengths
            stats['total_content_length'] += len(segment.content)
        
        if segments:
            stats['avg_segment_length'] = stats['total_content_length'] / len(segments)
        
        return stats