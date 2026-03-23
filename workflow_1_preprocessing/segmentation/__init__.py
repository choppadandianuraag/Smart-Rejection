"""
Section Segmentation Module
Segments resume/JD text into clearly defined sections using regex + heuristics.
Lightweight, no LLM required - runs on every document at ingest.
"""

import re
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class SectionType(Enum):
    """Resume section types."""
    CONTACT_INFO = "contact_info"
    SUMMARY = "summary"
    WORK_EXPERIENCE = "work_experience"
    EDUCATION = "education"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    OTHER = "other"


class JobSectionType(Enum):
    """Job description section types."""
    OVERVIEW = "overview"
    REQUIREMENTS = "requirements"
    RESPONSIBILITIES = "responsibilities"
    QUALIFICATIONS = "qualifications"
    NICE_TO_HAVE = "nice_to_have"
    OTHER = "other"


@dataclass
class Section:
    """Represents a segmented section of text."""
    section_type: str
    text: str
    char_offset_start: int
    char_offset_end: int
    section_order: int
    confidence_score: float
    
    def __repr__(self):
        return f"Section({self.section_type}, {len(self.text)} chars, conf={self.confidence_score:.2f})"


class ResumeSegmenter:
    """
    Segments resume text into sections using regex patterns and heuristics.
    Designed to be lightweight and fast.
    """
    
    # Section header patterns (case-insensitive)
    SECTION_PATTERNS = {
        SectionType.SUMMARY: [
            r'\b(professional\s+)?summary\b',
            r'\b(career\s+)?objective\b',
            r'\bprofile\b',
            r'\babout\s+me\b',
            r'\bpersonal\s+statement\b'
        ],
        SectionType.WORK_EXPERIENCE: [
            r'\b(work\s+)?experience\b',
            r'\bemployment(\s+history)?\b',
            r'\bprofessional\s+experience\b',
            r'\bwork\s+history\b',
            r'\bcareer\s+history\b'
        ],
        SectionType.EDUCATION: [
            r'\beducation\b',
            r'\bacademic\s+(background|qualifications)\b',
            r'\bqualifications\b',
            r'\bacademic\s+credentials\b'
        ],
        SectionType.SKILLS: [
            r'\b(technical\s+)?skills\b',
            r'\bcompetencies\b',
            r'\bexpertise\b',
            r'\bcore\s+skills\b',
            r'\bkey\s+skills\b',
            r'\btechnologies\b'
        ],
        SectionType.CERTIFICATIONS: [
            r'\bcertifications?\b',
            r'\blicen[cs]es?\b',
            r'\bprofessional\s+development\b',
            r'\bcredentials\b'
        ],
        SectionType.PROJECTS: [
            r'\bprojects?\b',
            r'\bportfolio\b',
            r'\bkey\s+projects\b',
            r'\bselected\s+projects\b'
        ]
    }
    
    # Confidence threshold for flagging manual review
    LOW_CONFIDENCE_THRESHOLD = 0.7
    
    def segment(self, raw_text: str) -> Tuple[List[Section], Dict[str, any]]:
        """
        Segment resume into sections.
        
        Args:
            raw_text: Full resume text
            
        Returns:
            Tuple of (sections list, metadata dict)
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty text provided for segmentation")
            return [], {"error": "Empty text"}
        
        sections = []
        lines = raw_text.split('\n')
        
        # Step 1: Extract contact info (first ~10-15 lines)
        contact_section, contact_end_idx = self._extract_contact_section(lines)
        if contact_section:
            sections.append(contact_section)
            lines = lines[contact_end_idx:]
            char_offset = len('\n'.join(raw_text.split('\n')[:contact_end_idx])) + 1
        else:
            char_offset = 0
        
        # Step 2: Detect section headers and segment
        current_section_type = None
        current_text_lines = []
        current_start_offset = char_offset
        section_order = 1 if contact_section else 0
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            if not line_clean:
                if current_text_lines:
                    current_text_lines.append(line)
                continue
            
            # Check if line is a section header
            detected_type, confidence = self._detect_section_header(line_clean)
            
            if detected_type and confidence > 0.6:
                # Save previous section
                if current_section_type and current_text_lines:
                    section_text = '\n'.join(current_text_lines).strip()
                    if section_text:
                        sections.append(Section(
                            section_type=current_section_type,
                            text=section_text,
                            char_offset_start=current_start_offset,
                            char_offset_end=current_start_offset + len(section_text),
                            section_order=section_order,
                            confidence_score=0.85
                        ))
                        section_order += 1
                        current_start_offset += len(section_text) + 1
                
                # Start new section
                current_section_type = detected_type
                current_text_lines = []
            else:
                # Add line to current section
                if current_section_type is None:
                    # Before first header - classify as OTHER
                    current_section_type = SectionType.OTHER.value
                current_text_lines.append(line)
        
        # Save final section
        if current_section_type and current_text_lines:
            section_text = '\n'.join(current_text_lines).strip()
            if section_text:
                sections.append(Section(
                    section_type=current_section_type,
                    text=section_text,
                    char_offset_start=current_start_offset,
                    char_offset_end=current_start_offset + len(section_text),
                    section_order=section_order,
                    confidence_score=0.85
                ))
        
        # Step 3: Calculate overall confidence and flag if needed
        metadata = self._calculate_segmentation_quality(sections, raw_text)
        
        logger.info(f"Segmented into {len(sections)} sections with avg confidence {metadata['avg_confidence']:.2f}")
        
        return sections, metadata
    
    def _extract_contact_section(self, lines: List[str]) -> Tuple[Optional[Section], int]:
        """
        Extract contact information section from top of resume.
        
        Returns:
            Tuple of (Section or None, end line index)
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b[\+]?[(]?\d{1,4}[)]?[-\s\.]?\d{1,4}[-\s\.]?\d{1,9}\b'
        
        contact_lines = []
        contact_end = 0
        found_email = False
        found_phone = False
        
        # Check first 15 lines
        for i, line in enumerate(lines[:15]):
            contact_lines.append(line)
            
            if re.search(email_pattern, line):
                found_email = True
            if re.search(phone_pattern, line):
                found_phone = True
            
            # Stop after finding both OR after a clear section header
            if (found_email or found_phone) and i > 3:
                if self._detect_section_header(line.strip())[1] > 0.7:
                    contact_end = i
                    break
                if i >= 10:  # Max 10 lines for contact
                    contact_end = i + 1
                    break
        
        if not (found_email or found_phone):
            return None, 0
        
        if contact_end == 0:
            contact_end = min(len(contact_lines), 10)
        
        contact_text = '\n'.join(contact_lines[:contact_end]).strip()
        confidence = 0.95 if (found_email and found_phone) else 0.8
        
        return Section(
            section_type=SectionType.CONTACT_INFO.value,
            text=contact_text,
            char_offset_start=0,
            char_offset_end=len(contact_text),
            section_order=0,
            confidence_score=confidence
        ), contact_end
    
    def _detect_section_header(self, line: str) -> Tuple[Optional[str], float]:
        """
        Detect if a line is a section header.
        
        Returns:
            Tuple of (section_type or None, confidence)
        """
        line_lower = line.lower()
        
        # Skip if line is too long (likely not a header)
        if len(line) > 60:
            return None, 0.0
        
        # Heuristics for headers
        is_all_caps = line.isupper() and len(line) > 2
        is_title_case = line.istitle()
        ends_with_colon = line.rstrip().endswith(':')
        is_bold_marker = line.startswith('**') or line.startswith('##')
        
        is_formatted_header = any([is_all_caps, is_title_case, ends_with_colon, is_bold_marker])
        
        # Match against known patterns
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    confidence = 0.9 if is_formatted_header else 0.7
                    return section_type.value, confidence
        
        return None, 0.0
    
    def _calculate_segmentation_quality(self, sections: List[Section], raw_text: str) -> Dict[str, any]:
        """Calculate quality metrics for segmentation."""
        if not sections:
            return {
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "section_count": 0,
                "needs_manual_review": True,
                "review_reason": "No sections detected"
            }
        
        confidences = [s.confidence_score for s in sections]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        
        # Flag for manual review if:
        # 1. Low overall confidence
        # 2. More than 30% sections are low confidence
        # 3. Missing critical sections (no experience or education)
        low_conf_count = sum(1 for c in confidences if c < self.LOW_CONFIDENCE_THRESHOLD)
        low_conf_ratio = low_conf_count / len(sections)
        
        section_types = {s.section_type for s in sections}
        has_experience = SectionType.WORK_EXPERIENCE.value in section_types
        has_education = SectionType.EDUCATION.value in section_types
        
        needs_review = False
        review_reason = None
        
        if avg_confidence < 0.75:
            needs_review = True
            review_reason = "Low overall segmentation confidence"
        elif low_conf_ratio > 0.3:
            needs_review = True
            review_reason = f"{int(low_conf_ratio*100)}% sections have low confidence"
        elif not (has_experience or has_education):
            needs_review = True
            review_reason = "Missing critical sections (experience/education)"
        
        return {
            "avg_confidence": round(avg_confidence, 3),
            "min_confidence": round(min_confidence, 3),
            "section_count": len(sections),
            "section_types": list(section_types),
            "needs_manual_review": needs_review,
            "review_reason": review_reason
        }


class JobDescriptionSegmenter:
    """
    Segments job description text into sections.
    """
    
    SECTION_PATTERNS = {
        JobSectionType.OVERVIEW: [
            r'\babout(\s+the)?\s+(company|role|position|job)\b',
            r'\boverview\b',
            r'\bjob\s+summary\b',
            r'\bposition\s+summary\b'
        ],
        JobSectionType.REQUIREMENTS: [
            r'\brequirements?\b',
            r'\bqualifications?\b',
            r'\bmust\s+have\b',
            r'\brequired\s+skills\b',
            r'\bminimum\s+qualifications\b'
        ],
        JobSectionType.RESPONSIBILITIES: [
            r'\bresponsibilities\b',
            r'\bduties\b',
            r'\bwhat\s+you\'?ll\s+do\b',
            r'\byour\s+role\b',
            r'\bkey\s+responsibilities\b'
        ],
        JobSectionType.QUALIFICATIONS: [
            r'\bpreferred\s+qualifications?\b',
            r'\bdesired\s+skills\b',
            r'\bideal\s+candidate\b'
        ],
        JobSectionType.NICE_TO_HAVE: [
            r'\bnice\s+to\s+have\b',
            r'\bbonus\s+points?\b',
            r'\bplus(es)?\b',
            r'\badditional\s+skills\b'
        ]
    }
    
    def segment(self, job_description: str) -> Tuple[List[Section], Dict[str, any]]:
        """
        Segment job description into sections.
        Similar logic to resume segmentation but with JD-specific patterns.
        """
        sections = []
        lines = job_description.split('\n')
        
        current_section_type = JobSectionType.OVERVIEW.value  # Default to overview
        current_text_lines = []
        current_start_offset = 0
        section_order = 0
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            if not line_clean:
                if current_text_lines:
                    current_text_lines.append(line)
                continue
            
            detected_type, confidence = self._detect_section_header(line_clean)
            
            if detected_type and confidence > 0.6:
                # Save previous section
                if current_text_lines:
                    section_text = '\n'.join(current_text_lines).strip()
                    if section_text:
                        sections.append(Section(
                            section_type=current_section_type,
                            text=section_text,
                            char_offset_start=current_start_offset,
                            char_offset_end=current_start_offset + len(section_text),
                            section_order=section_order,
                            confidence_score=0.85
                        ))
                        section_order += 1
                        current_start_offset += len(section_text) + 1
                
                current_section_type = detected_type
                current_text_lines = []
            else:
                current_text_lines.append(line)
        
        # Save final section
        if current_text_lines:
            section_text = '\n'.join(current_text_lines).strip()
            if section_text:
                sections.append(Section(
                    section_type=current_section_type,
                    text=section_text,
                    char_offset_start=current_start_offset,
                    char_offset_end=current_start_offset + len(section_text),
                    section_order=section_order,
                    confidence_score=0.85
                ))
        
        metadata = {
            "section_count": len(sections),
            "section_types": [s.section_type for s in sections]
        }
        
        logger.info(f"Segmented JD into {len(sections)} sections")
        
        return sections, metadata
    
    def _detect_section_header(self, line: str) -> Tuple[Optional[str], float]:
        """Detect job description section headers."""
        line_lower = line.lower()
        
        if len(line) > 60:
            return None, 0.0
        
        is_formatted = (
            line.isupper() or 
            line.istitle() or 
            line.endswith(':') or
            line.startswith('**') or
            line.startswith('##')
        )
        
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    confidence = 0.9 if is_formatted else 0.7
                    return section_type.value, confidence
        
        return None, 0.0


# Factory functions
def segment_resume(raw_text: str) -> Tuple[List[Section], Dict[str, any]]:
    """Convenience function to segment a resume."""
    segmenter = ResumeSegmenter()
    return segmenter.segment(raw_text)


def segment_job_description(job_text: str) -> Tuple[List[Section], Dict[str, any]]:
    """Convenience function to segment a job description."""
    segmenter = JobDescriptionSegmenter()
    return segmenter.segment(job_text)
