"""
View all resume data from Supabase in a clear format.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from database.supabase_client import SupabaseClient


def main():
    client = SupabaseClient().client
    result = client.table('resumes').select('*').execute()
    
    if not result.data:
        print("No data found in resumes table")
        return
    
    for row in result.data:
        print("=" * 70)
        print("                    RESUME DATABASE RECORD")
        print("=" * 70)
        print()
        print("COLUMN                  | VALUE")
        print("-" * 70)
        print(f"id                      | {row.get('id')}")
        print(f"filename                | {row.get('filename')}")
        print(f"file_type               | {row.get('file_type')}")
        print(f"file_size_bytes         | {row.get('file_size_bytes')}")
        print(f"processing_status       | {row.get('processing_status')}")
        print(f"embedding_model         | {row.get('embedding_model')}")
        print(f"embedding_vector        | {row.get('embedding_vector')}")
        print(f"error_message           | {row.get('error_message')}")
        print(f"created_at              | {row.get('created_at')}")
        print(f"updated_at              | {row.get('updated_at')}")
        print()
        
        print("-" * 70)
        print("EXTRACTED DATA (JSONB column)")
        print("-" * 70)
        extracted = row.get('extracted_data', {})
        print(json.dumps(extracted, indent=2))
        print()
        
        print("-" * 70)
        print("METADATA (JSONB column)")
        print("-" * 70)
        metadata = row.get('metadata', {})
        print(json.dumps(metadata, indent=2))
        print()
        
        print("-" * 70)
        print("RAW_TEXT (TEXT column) - FULL CONTENT")
        print("-" * 70)
        raw_text = row.get('raw_text', '')
        print(raw_text if raw_text else "None")
        print()
        
        print("-" * 70)
        print("MARKDOWN_CONTENT (TEXT column) - FULL CONTENT")
        print("-" * 70)
        markdown = row.get('markdown_content', '')
        print(markdown if markdown else "None")
        print()
        print("=" * 70)


if __name__ == "__main__":
    main()
