"""
Unified Database Queries for DreamLayer
Provides consistent database access across all components
"""

import os
import sys
from typing import List, Dict, Any, Optional

def get_database_connection():
    """Get database connection using consistent path"""
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'scripts'))
    from database import DreamLayerDB
    return DreamLayerDB()

def get_all_runs_with_clipscore() -> List[Dict[str, Any]]:
    """
    Unified query to get all runs with ClipScore
    Used by: run_registry.py, report_bundle.py, frontend APIs
    """
    try:
        db = get_database_connection()
        
        with db.get_connection() as conn:
            # Single unified query that joins runs with metrics
            cursor = conn.execute("""
                SELECT 
                    r.*,
                    m.metric_value as clip_score_mean
                FROM runs r
                LEFT JOIN metrics m ON r.run_id = m.run_id AND m.metric_name = 'clip_score_mean'
                ORDER BY r.timestamp DESC
            """)
            
            runs = []
            for row in cursor.fetchall():
                run_dict = dict(row)
                # Convert JSON strings back to objects and ensure arrays exist
                for field in ['loras', 'controlnets', 'workflow']:
                    if run_dict.get(field):
                        try:
                            import json
                            run_dict[field] = json.loads(run_dict[field])
                        except (json.JSONDecodeError, TypeError):
                            run_dict[field] = [] if field != 'workflow' else {}
                    else:
                        run_dict[field] = [] if field != 'workflow' else {}
                
                # Ensure generated_images is an array
                if not run_dict.get('generated_images'):
                    run_dict['generated_images'] = []
                elif isinstance(run_dict['generated_images'], str):
                    try:
                        import json
                        run_dict['generated_images'] = json.loads(run_dict['generated_images'])
                    except (json.JSONDecodeError, TypeError):
                        # If it's a single filename string, convert to array
                        run_dict['generated_images'] = [run_dict['generated_images']] if run_dict['generated_images'] else []
                
                # Ensure it's always a list
                if not isinstance(run_dict['generated_images'], list):
                    run_dict['generated_images'] = []
                
                runs.append(run_dict)
            
            return runs
            
    except Exception as e:
        print(f"Error getting runs with ClipScore: {e}")
        return []

def get_single_run_with_clipscore(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Unified query to get single run with ClipScore
    Used by: run_registry.py, report_bundle.py
    """
    try:
        db = get_database_connection()
        
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    r.*,
                    m.metric_value as clip_score_mean
                FROM runs r
                LEFT JOIN metrics m ON r.run_id = m.run_id AND m.metric_name = 'clip_score_mean'
                WHERE r.run_id = ?
            """, (run_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            run_dict = dict(row)
            # Convert JSON strings back to objects and ensure arrays exist
            for field in ['loras', 'controlnets', 'workflow']:
                if run_dict.get(field):
                    try:
                        import json
                        run_dict[field] = json.loads(run_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        run_dict[field] = [] if field != 'workflow' else {}
                else:
                    run_dict[field] = [] if field != 'workflow' else {}
            
            # Ensure generated_images is an array
            if not run_dict.get('generated_images'):
                run_dict['generated_images'] = []
            elif isinstance(run_dict['generated_images'], str):
                try:
                    import json
                    run_dict['generated_images'] = json.loads(run_dict['generated_images'])
                except (json.JSONDecodeError, TypeError):
                    run_dict['generated_images'] = [run_dict['generated_images']] if run_dict['generated_images'] else []
            
            return run_dict
            
    except Exception as e:
        print(f"Error getting run {run_id} with ClipScore: {e}")
        return None

def save_run_to_database(run_data: Dict[str, Any]) -> bool:
    """
    Unified function to save run to database
    Used by: txt2img_server.py, img2img_server.py, run_registry.py
    """
    try:
        db = get_database_connection()
        
        # Convert arrays to JSON strings for database storage
        run_data_copy = run_data.copy()
        for field in ['loras', 'controlnets', 'workflow', 'generated_images']:
            if field in run_data_copy and run_data_copy[field]:
                import json
                run_data_copy[field] = json.dumps(run_data_copy[field])
        
        return db.insert_run(run_data_copy)
    except Exception as e:
        print(f"Error saving run to database: {e}")
        return False

def save_assets_to_database(run_id: str, timestamp: str, generated_images: List[str]) -> bool:
    """
    Unified function to save assets to database
    Used by: txt2img_server.py, img2img_server.py, run_registry.py
    """
    try:
        db = get_database_connection()
        output_dir = os.path.join('..', '..', 'Dream_Layer_Resources', 'output')
        return db.insert_assets(run_id, timestamp, generated_images, output_dir)
    except Exception as e:
        print(f"Error saving assets to database: {e}")
        return False

def save_clipscore_to_database(run_id: str, timestamp: str, clip_score: float) -> bool:
    """
    Unified function to save ClipScore to database
    Used by: ClipScore calculation modules
    """
    try:
        db = get_database_connection()
        
        with db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metrics (
                    run_id, timestamp, metric_name, metric_value
                ) VALUES (?, ?, ?, ?)
            """, (run_id, timestamp, 'clip_score_mean', clip_score))
            conn.commit()
        
        return True
    except Exception as e:
        print(f"Error saving ClipScore to database: {e}")
        return False

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    try:
        db = get_database_connection()
        
        with db.get_connection() as conn:
            # Count runs
            cursor = conn.execute("SELECT COUNT(*) as count FROM runs")
            total_runs = cursor.fetchone()['count']
            
            # Count runs with ClipScore
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM runs r
                INNER JOIN metrics m ON r.run_id = m.run_id 
                WHERE m.metric_name = 'clip_score_mean'
            """)
            runs_with_clipscore = cursor.fetchone()['count']
            
            # Count total assets
            cursor = conn.execute("SELECT COUNT(*) as count FROM assets")
            total_assets = cursor.fetchone()['count']
            
            return {
                'database_path': db.db_path,
                'total_runs': total_runs,
                'runs_with_clipscore': runs_with_clipscore,
                'total_assets': total_assets,
                'runs_without_clipscore': total_runs - runs_with_clipscore
            }
            
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Test unified queries
    print("🧪 Testing Unified Database Queries")
    print("=" * 40)
    
    # Test database stats
    stats = get_database_stats()
    print(f"Database stats: {stats}")
    
    # Test getting all runs
    runs = get_all_runs_with_clipscore()
    print(f"Total runs with ClipScore: {len(runs)}")
    
    if runs:
        sample = runs[0]
        print(f"Sample run: {sample['run_id'][:8]}... ClipScore: {sample.get('clip_score_mean', 'None')}")
