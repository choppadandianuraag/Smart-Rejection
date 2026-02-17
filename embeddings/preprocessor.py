"""
Text Preprocessor for Resume Data.
Cleans and normalizes resume text for better embedding quality.
"""

import re
from typing import List, Dict, Optional
from loguru import logger


class ResumePreprocessor:
    """
    Preprocesses resume text by cleaning, normalizing, and standardizing.
    """
    
    # Common company name variations to normalize
    COMPANY_NORMALIZATIONS = {
        r'\bGoogle\s*(?:Inc\.?|LLC|Corp\.?)?\b': 'Google',
        r'\bMicrosoft\s*(?:Corporation|Corp\.?)?\b': 'Microsoft',
        r'\bAmazon\s*(?:\.com|Inc\.?|Web Services|AWS)?\b': 'Amazon',
        r'\bMeta\s*(?:Platforms|Inc\.?)?\b|\bFacebook\s*(?:Inc\.?)?\b': 'Meta',
        r'\bApple\s*(?:Inc\.?)?\b': 'Apple',
        r'\bIBM\s*(?:Corporation|Corp\.?)?\b': 'IBM',
        r'\bNetflix\s*(?:Inc\.?)?\b': 'Netflix',
        r'\bUber\s*(?:Technologies|Inc\.?)?\b': 'Uber',
        r'\bAirbnb\s*(?:Inc\.?)?\b': 'Airbnb',
        r'\bDeloitte\s*(?:LLP|Touche)?\b|\bDelloite\b': 'Deloitte',
        r'\bAccenture\s*(?:PLC)?\b': 'Accenture',
        r'\bInfosys\s*(?:Limited|Ltd\.?)?\b': 'Infosys',
        r'\bTCS\b|\bTata\s*Consultancy\s*Services\b': 'TCS',
        r'\bWipro\s*(?:Limited|Ltd\.?)?\b': 'Wipro',
    }
    
    # Job title normalizations
    TITLE_NORMALIZATIONS = {
        r'\bSr\.?\s*': 'Senior ',
        r'\bJr\.?\s*': 'Junior ',
        r'\bSoftware\s*Eng\.?\b': 'Software Engineer',
        r'\bSDE\b': 'Software Development Engineer',
        r'\bSWE\b': 'Software Engineer',
        r'\bML\s*Eng(?:ineer)?\b': 'Machine Learning Engineer',
        r'\bData\s*Sci(?:entist)?\b': 'Data Scientist',
        r'\bDevOps\s*Eng(?:ineer)?\b': 'DevOps Engineer',
        r'\bFull\s*Stack\s*Dev(?:eloper)?\b': 'Full Stack Developer',
        r'\bFrontend\s*Dev(?:eloper)?\b': 'Frontend Developer',
        r'\bBackend\s*Dev(?:eloper)?\b': 'Backend Developer',
        r'\bPM\b': 'Product Manager',
        r'\bTPM\b': 'Technical Program Manager',
    }
    
    # Skill normalizations
    SKILL_NORMALIZATIONS = {
        r'\bJS\b': 'JavaScript',
        r'\bTS\b': 'TypeScript',
        r'\bPy\b|\bPython3?\b': 'Python',
        r'\bReact\.?js\b|\bReactJS\b': 'React',
        r'\bNode\.?js\b|\bNodeJS\b': 'Node.js',
        r'\bVue\.?js\b|\bVueJS\b': 'Vue.js',
        r'\bAngular\.?js\b|\bAngularJS\b': 'Angular',
        r'\bTensorflow\b': 'TensorFlow',
        r'\bPytorch\b': 'PyTorch',
        r'\bK8s\b': 'Kubernetes',
        r'\bPostgres\b': 'PostgreSQL',
        r'\bMongo\b': 'MongoDB',
        r'\bAWS\b': 'Amazon Web Services',
        r'\bGCP\b': 'Google Cloud Platform',
        r'\bML\b': 'Machine Learning',
        r'\bDL\b': 'Deep Learning',
        r'\bNLP\b': 'Natural Language Processing',
        r'\bCV\b': 'Computer Vision',
        r'\bCI/CD\b|\bCI\s*/\s*CD\b': 'CI/CD',
        r'\bAPI\b': 'API',
        r'\bREST\b|\bRESTful\b': 'REST API',
    }
    
    # Date patterns to standardize
    DATE_PATTERNS = [
        # "Jan 2023" -> "January 2023"
        (r'\bJan\b', 'January'),
        (r'\bFeb\b', 'February'),
        (r'\bMar\b', 'March'),
        (r'\bApr\b', 'April'),
        (r'\bJun\b', 'June'),
        (r'\bJul\b', 'July'),
        (r'\bAug\b', 'August'),
        (r'\bSep(?:t)?\b', 'September'),
        (r'\bOct\b', 'October'),
        (r'\bNov\b', 'November'),
        (r'\bDec\b', 'December'),
    ]
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to the text.
        
        Args:
            text: Raw resume text.
            
        Returns:
            Preprocessed and cleaned text.
        """
        if not text:
            return ""
        
        # Step 1: Basic cleaning
        text = self.clean_text(text)
        
        # Step 2: Standardize dates
        text = self.standardize_dates(text)
        
        # Step 3: Normalize company names
        text = self.normalize_companies(text)
        
        # Step 4: Normalize job titles
        text = self.normalize_titles(text)
        
        # Step 5: Normalize skills
        text = self.normalize_skills(text)
        
        # Step 6: Final cleanup
        text = self.final_cleanup(text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw text.
            
        Returns:
            Cleaned text.
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses (but keep the fact there was one)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', 'EMAIL', text)
        
        # Remove phone numbers (but keep the fact there was one)
        text = re.sub(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', 'PHONE', text)
        
        # Remove special unicode characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Remove bullet points and common list markers
        text = re.sub(r'^[\s]*[•●○◦▪▫‣⁃◆◇►▸➤➢→⮞]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'[•●○◦▪▫‣⁃◆◇►▸➤➢→⮞]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[-]{3,}', ' ', text)
        
        return text.strip()
    
    def standardize_dates(self, text: str) -> str:
        """
        Standardize date formats.
        
        Args:
            text: Text with various date formats.
            
        Returns:
            Text with standardized dates.
        """
        # Expand abbreviated months
        for pattern, replacement in self.DATE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Standardize date ranges
        # "2020 - Present" -> "2020 to Present"
        text = re.sub(r'(\d{4})\s*[-–—]\s*(Present|Current|Now|\d{4})', r'\1 to \2', text, flags=re.IGNORECASE)
        
        # "January 2020 - Present" -> "January 2020 to Present"
        text = re.sub(r'(\w+\s+\d{4})\s*[-–—]\s*(Present|Current|Now|\w+\s+\d{4})', r'\1 to \2', text, flags=re.IGNORECASE)
        
        # Standardize "Present/Current/Now" to "Present"
        text = re.sub(r'\b(Current|Now)\b', 'Present', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_companies(self, text: str) -> str:
        """
        Normalize company name variations.
        
        Args:
            text: Text with company names.
            
        Returns:
            Text with normalized company names.
        """
        for pattern, replacement in self.COMPANY_NORMALIZATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_titles(self, text: str) -> str:
        """
        Normalize job title variations.
        
        Args:
            text: Text with job titles.
            
        Returns:
            Text with normalized titles.
        """
        for pattern, replacement in self.TITLE_NORMALIZATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_skills(self, text: str) -> str:
        """
        Normalize skill name variations.
        
        Args:
            text: Text with skill names.
            
        Returns:
            Text with normalized skill names.
        """
        for pattern, replacement in self.SKILL_NORMALIZATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def final_cleanup(self, text: str) -> str:
        """
        Final cleanup pass.
        
        Args:
            text: Preprocessed text.
            
        Returns:
            Final cleaned text.
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        # Ensure proper sentence spacing
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        
        return text.strip()
    
    def extract_skills_list(self, text: str) -> List[str]:
        """
        Extract a list of normalized skills from text.
        
        Args:
            text: Resume text.
            
        Returns:
            List of extracted skills.
        """
        # Common technical skills to look for
        skill_patterns = [
            r'\bPython\b', r'\bJava\b', r'\bJavaScript\b', r'\bTypeScript\b',
            r'\bC\+\+\b', r'\bC#\b', r'\bGo\b', r'\bRust\b', r'\bRuby\b',
            r'\bPHP\b', r'\bSwift\b', r'\bKotlin\b', r'\bScala\b', r'\bR\b',
            r'\bSQL\b', r'\bNoSQL\b', r'\bMongoDB\b', r'\bPostgreSQL\b', r'\bMySQL\b',
            r'\bRedis\b', r'\bElasticsearch\b', r'\bKafka\b', r'\bRabbitMQ\b',
            r'\bDocker\b', r'\bKubernetes\b', r'\bAWS\b', r'\bAzure\b', r'\bGCP\b',
            r'\bTerraform\b', r'\bAnsible\b', r'\bJenkins\b', r'\bGitHub Actions\b',
            r'\bReact\b', r'\bAngular\b', r'\bVue\.js\b', r'\bNode\.js\b',
            r'\bDjango\b', r'\bFlask\b', r'\bFastAPI\b', r'\bSpring\b',
            r'\bTensorFlow\b', r'\bPyTorch\b', r'\bKeras\b', r'\bScikit-learn\b',
            r'\bPandas\b', r'\bNumPy\b', r'\bOpenCV\b', r'\bNLTK\b', r'\bSpaCy\b',
            r'\bMachine Learning\b', r'\bDeep Learning\b', r'\bNLP\b',
            r'\bComputer Vision\b', r'\bData Science\b', r'\bData Analysis\b',
            r'\bAgile\b', r'\bScrum\b', r'\bCI/CD\b', r'\bDevOps\b', r'\bMLOps\b',
            r'\bGit\b', r'\bLinux\b', r'\bREST API\b', r'\bGraphQL\b',
            r'\bLangChain\b', r'\bRAG\b', r'\bLLM\b', r'\bGPT\b', r'\bBERT\b',
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend([m.strip() for m in matches])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills


def preprocess_resume(text: str) -> str:
    """
    Convenience function to preprocess resume text.
    
    Args:
        text: Raw resume text.
        
    Returns:
        Preprocessed text.
    """
    preprocessor = ResumePreprocessor()
    return preprocessor.preprocess(text)
