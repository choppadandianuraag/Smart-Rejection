"""
Supabase database operations for Smart Rejection system.
Handles all database interactions with Supabase.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from supabase import create_client, Client
from loguru import logger

from config.settings import settings
from database.models import Resume, ResumeCreate


class SupabaseClient:
    """Singleton Supabase client for database operations."""
    
    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
            logger.info("Supabase client initialized successfully")
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance."""
        return self._client
    
    def insert_resume(self, resume_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert a resume into the database.
        
        Args:
            resume_data: Dictionary with resume fields
            
        Returns:
            Inserted resume data with ID, or None on failure
        """
        try:
            resume_id = str(uuid4())
            now = datetime.utcnow().isoformat()
            
            data = {
                "id": resume_id,
                "filename": resume_data.get("filename"),
                "file_type": resume_data.get("file_type"),
                "file_size_bytes": resume_data.get("file_size_bytes"),
                "raw_text": resume_data.get("raw_text"),
                "markdown_content": resume_data.get("markdown_content", resume_data.get("raw_text", "")),
                "metadata": resume_data.get("metadata", {}),
                "processing_status": resume_data.get("status", "processed"),
                "created_at": now,
                "updated_at": now
            }
            
            result = self._client.table("resumes").insert(data).execute()
            
            if result.data:
                logger.info(f"Resume inserted with ID: {resume_id}")
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error inserting resume: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            # Count total resumes
            resumes = self._client.table("resumes").select("id", count="exact").execute()
            total_resumes = resumes.count if resumes.count else len(resumes.data)
            
            # Count resumes with embeddings
            with_embeddings = self._client.table("resumes").select("id", count="exact").not_.is_("embedding_vector", "null").execute()
            total_with_embeddings = with_embeddings.count if with_embeddings.count else len(with_embeddings.data)
            
            return {
                "total_resumes": total_resumes,
                "resumes_with_embeddings": total_with_embeddings
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}


class ResumeRepository:
    """Repository for resume database operations."""
    
    TABLE_NAME = "resumes"
    
    def __init__(self):
        self.db = SupabaseClient().client
    
    async def create(self, resume_data: ResumeCreate) -> Resume:
        """
        Create a new resume entry in the database.
        
        Args:
            resume_data: Resume data to insert
            
        Returns:
            Created resume with generated ID
        """
        try:
            resume_id = str(uuid4())
            now = datetime.utcnow().isoformat()
            
            data = {
                "id": resume_id,
                "filename": resume_data.filename,
                "file_type": resume_data.file_type,
                "file_size_bytes": resume_data.file_size_bytes,
                "raw_text": resume_data.raw_text,
                "markdown_content": resume_data.markdown_content,
                "extracted_data": resume_data.extracted_data,
                "metadata": resume_data.metadata,
                "processing_status": "completed",
                "created_at": now,
                "updated_at": now
            }
            
            result = self.db.table(self.TABLE_NAME).insert(data).execute()
            
            if result.data:
                logger.success(f"Resume created with ID: {resume_id}")
                return Resume(**result.data[0])
            else:
                raise Exception("Failed to create resume: No data returned")
                
        except Exception as e:
            logger.error(f"Error creating resume: {str(e)}")
            raise
    
    def create_sync(self, resume_data: ResumeCreate) -> Resume:
        """
        Synchronous version of create method.
        
        Args:
            resume_data: Resume data to insert
            
        Returns:
            Created resume with generated ID
        """
        try:
            resume_id = str(uuid4())
            now = datetime.utcnow().isoformat()
            
            data = {
                "id": resume_id,
                "filename": resume_data.filename,
                "file_type": resume_data.file_type,
                "file_size_bytes": resume_data.file_size_bytes,
                "raw_text": resume_data.raw_text,
                "markdown_content": resume_data.markdown_content,
                "extracted_data": resume_data.extracted_data,
                "metadata": resume_data.metadata,
                "processing_status": "completed",
                "created_at": now,
                "updated_at": now
            }
            
            result = self.db.table(self.TABLE_NAME).insert(data).execute()
            
            if result.data:
                logger.success(f"Resume created with ID: {resume_id}")
                return Resume(**result.data[0])
            else:
                raise Exception("Failed to create resume: No data returned")
                
        except Exception as e:
            logger.error(f"Error creating resume: {str(e)}")
            raise
    
    def get_by_id(self, resume_id: str) -> Optional[Resume]:
        """
        Retrieve a resume by its ID.
        
        Args:
            resume_id: UUID of the resume
            
        Returns:
            Resume if found, None otherwise
        """
        try:
            result = self.db.table(self.TABLE_NAME)\
                .select("*")\
                .eq("id", resume_id)\
                .execute()
            
            if result.data:
                return Resume(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving resume {resume_id}: {str(e)}")
            raise
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Resume]:
        """
        Retrieve all resumes with pagination.
        
        Args:
            limit: Maximum number of resumes to return
            offset: Number of resumes to skip
            
        Returns:
            List of resumes
        """
        try:
            result = self.db.table(self.TABLE_NAME)\
                .select("*")\
                .order("created_at", desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            return [Resume(**item) for item in result.data]
            
        except Exception as e:
            logger.error(f"Error retrieving resumes: {str(e)}")
            raise
    
    def update(self, resume_id: str, updates: Dict[str, Any]) -> Resume:
        """
        Update a resume entry.
        
        Args:
            resume_id: UUID of the resume to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated resume
        """
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.db.table(self.TABLE_NAME)\
                .update(updates)\
                .eq("id", resume_id)\
                .execute()
            
            if result.data:
                logger.info(f"Resume {resume_id} updated successfully")
                return Resume(**result.data[0])
            else:
                raise Exception(f"Resume {resume_id} not found")
                
        except Exception as e:
            logger.error(f"Error updating resume {resume_id}: {str(e)}")
            raise
    
    def update_embedding(
        self, 
        resume_id: str, 
        embedding_vector: List[float],
        embedding_model: str
    ) -> Resume:
        """
        Update the embedding vector for a resume.
        
        Args:
            resume_id: UUID of the resume
            embedding_vector: The embedding vector
            embedding_model: Name of the model used
            
        Returns:
            Updated resume
        """
        return self.update(resume_id, {
            "embedding_vector": embedding_vector,
            "embedding_model": embedding_model
        })
    
    def delete(self, resume_id: str) -> bool:
        """
        Delete a resume by its ID.
        
        Args:
            resume_id: UUID of the resume to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.db.table(self.TABLE_NAME)\
                .delete()\
                .eq("id", resume_id)\
                .execute()
            
            logger.info(f"Resume {resume_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting resume {resume_id}: {str(e)}")
            raise
    
    def search_by_filename(self, filename: str) -> List[Resume]:
        """
        Search resumes by filename pattern.
        
        Args:
            filename: Filename pattern to search
            
        Returns:
            List of matching resumes
        """
        try:
            result = self.db.table(self.TABLE_NAME)\
                .select("*")\
                .ilike("filename", f"%{filename}%")\
                .execute()
            
            return [Resume(**item) for item in result.data]
            
        except Exception as e:
            logger.error(f"Error searching resumes: {str(e)}")
            raise
    
    def get_by_status(self, status: str) -> List[Resume]:
        """
        Get resumes by processing status.
        
        Args:
            status: Processing status (pending, processing, completed, failed)
            
        Returns:
            List of resumes with the given status
        """
        try:
            result = self.db.table(self.TABLE_NAME)\
                .select("*")\
                .eq("processing_status", status)\
                .execute()
            
            return [Resume(**item) for item in result.data]
            
        except Exception as e:
            logger.error(f"Error retrieving resumes by status: {str(e)}")
            raise


def get_resume_repository() -> ResumeRepository:
    """Get a resume repository instance."""
    return ResumeRepository()
