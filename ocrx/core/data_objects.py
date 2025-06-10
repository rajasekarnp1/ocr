from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
import numpy as np # For type hinting, will be a core dependency

@dataclass
class TextRegion:
    region_id: str # Unique identifier for the region on its page
    image_crop: Optional[np.ndarray] = None # Image data of the segmented region
    bounding_box: Tuple[int, int, int, int] # (x_min, y_min, x_max, y_max)
    region_type: str = "text_line" # e.g., "text_line", "paragraph", "table_cell", "figure_caption"
    sequence_id: int = 0 # For reading order
    # Add other potential attributes like orientation, skew angle if determined per region
    raw_ocr_results: List['RecognitionResult'] = field(default_factory=list)
    consensus_candidate: Optional['MainCandidate'] = None
    postprocessed_candidate: Optional['MainCandidate'] = None


@dataclass
class RecognitionResult:
    text: str
    confidence: float # Overall confidence for this hypothesis
    char_confidences: Optional[List[float]] = None
    char_boxes: Optional[List[Tuple[int, int, int, int]]] = None # Relative to region crop
    word_confidences: Optional[List[float]] = None # If engine provides word-level
    word_boxes: Optional[List[Tuple[int, int, int, int]]] = None # Relative to region crop
    engine_id: Optional[str] = None


@dataclass
class MainCandidate:
    text: str
    confidence: float # Aggregated/final confidence
    source_engines: List[str] = field(default_factory=list) # Engines contributing to this
    # Potentially, a list of alternative texts if needed later
    # alternatives: List[RecognitionResult] = field(default_factory=list)


@dataclass
class PageContext:
    page_number: int
    original_image: Optional[np.ndarray] = None # Keep if needed for output generation
    preprocessed_image: Optional[np.ndarray] = None
    layout_regions: List[TextRegion] = field(default_factory=list)
    # Errors specific to this page
    errors: List[str] = field(default_factory=list)
    processing_times: Dict[str, float] = field(default_factory=dict) # e.g., {"preprocessing": 0.5, "layout": 1.2}


@dataclass
class DocumentContext:
    document_id: str # e.g., filename or unique ID
    source_path_or_id: str
    global_config: Dict[str, Any] = field(default_factory=dict) # Runtime config for this doc
    pages: List[PageContext] = field(default_factory=list)
    overall_status: str = "pending" # e.g., "pending", "processing", "completed", "failed"
    # Errors pertaining to the whole document processing
    document_errors: List[str] = field(default_factory=list)
    total_processing_time: Optional[float] = None
