#!/usr/bin/env python3
"""
Real wandb data parser using wandb's native functionality.

This script uses wandb's built-in functions to properly parse .wandb files
and extract the actual training metrics.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings


def parse_wandb_file_native(wandb_file: str) -> Optional[pd.DataFrame]:
    """
    Parse a wandb file using wandb's native functionality.
    
    Args:
        wandb_file: Path to the .wandb file
        
    Returns:
        DataFrame with training metrics or None if parsing fails
    """
    try:
        import wandb
        
        # Get the run directory
        run_dir = Path(wandb_file).parent
        
        # Try to use wandb.restore() to get the run data
        print(f"Attempting to restore wandb run from {run_dir}")
        
        # Method 1: Try using wandb.restore()
        try:
            # Change to the run directory
            original_cwd = os.getcwd()
            os.chdir(run_dir)
            
            # Initialize wandb in offline mode
            os.environ["WANDB_MODE"] = "offline"
            
            # Try to restore the run
            restored_run = wandb.restore(str(wandb_file))
            print(f"Successfully restored run: {restored_run}")
            
            # Get history data
            if hasattr(restored_run, 'history'):
                history_df = restored_run.history()
                if not history_df.empty:
                    print(f"Found history data with shape: {history_df.shape}")
                    print(f"Columns: {list(history_df.columns)}")
                    return history_df
            
        except Exception as e:
            print(f"wandb.restore() failed: {e}")
        finally:
            os.chdir(original_cwd)
        
        # Method 2: Try using wandb.Api() with local files
        try:
            api = wandb.Api()
            
            # Extract run ID from filename
            run_id = Path(wandb_file).stem.replace("run-", "")
            print(f"Trying to access run with ID: {run_id}")
            
            # This might not work for offline runs, but worth trying
            run = api.run(f"local/{run_id}")
            history_df = run.history()
            
            if not history_df.empty:
                print(f"Found API history data with shape: {history_df.shape}")
                return history_df
                
        except Exception as e:
            print(f"wandb.Api() approach failed: {e}")
        
        # Method 3: Try to read the binary file directly using wandb internals
        try:
            # Import wandb internal modules
            from wandb.sdk.internal import datastore
            from wandb.sdk.lib import rundata
            
            print("Trying wandb internal datastore...")
            
            # Try to read using datastore
            ds = datastore.DataStore()
            ds.open_for_read(str(wandb_file))
            
            # Read all records
            records = []
            while True:
                try:
                    record = ds.read_record()
                    if record is None:
                        break
                    records.append(record)
                except Exception:
                    break
            
            print(f"Read {len(records)} records from datastore")
            
            # Parse records to extract metrics
            metrics_data = []
            for record in records:
                if hasattr(record, 'history') and record.history:
                    history_item = record.history
                    if hasattr(history_item, 'item') and history_item.item:
                        metrics_data.append(json.loads(history_item.item))
                elif hasattr(record, 'summary') and record.summary:
                    summary_item = record.summary
                    if hasattr(summary_item, 'update') and summary_item.update:
                        metrics_data.append(json.loads(summary_item.update))
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                print(f"Extracted {len(metrics_data)} metric records")
                print(f"Columns: {list(df.columns)}")
                return df
                
        except ImportError as e:
            print(f"wandb internal modules not available: {e}")
        except Exception as e:
            print(f"wandb internal parsing failed: {e}")
        
        # Method 4: Try using protobuf parsing (wandb uses protobuf internally)
        try:
            print("Trying protobuf parsing...")
            
            # Read the file as binary
            with open(wandb_file, 'rb') as f:
                content = f.read()
            
            # Try to find JSON-like data in the binary content
            # wandb files often contain JSON data embedded in the binary
            import re
            
            # Look for JSON patterns in the binary data
            json_pattern = rb'\{[^}]*"_step"[^}]*\}'
            matches = re.findall(json_pattern, content)
            
            if matches:
                print(f"Found {len(matches)} potential JSON records")
                
                metrics_data = []
                for match in matches:
                    try:
                        # Decode and parse JSON
                        json_str = match.decode('utf-8', errors='ignore')
                        data = json.loads(json_str)
                        metrics_data.append(data)
                    except:
                        continue
                
                if metrics_data:
                    df = pd.DataFrame(metrics_data)
                    print(f"Successfully parsed {len(metrics_data)} records")
                    print(f"Columns: {list(df.columns)}")
                    return df
            
        except Exception as e:
            print(f"Protobuf/JSON parsing failed: {e}")
        
        print("All parsing methods failed")
        return None
        
    except ImportError:
        print("wandb library not available")
        return None


def extract_real_wandb_data(run_dir: str) -> Optional[pd.DataFrame]:
    """
    Extract real data from a wandb run directory.
    
    Args:
        run_dir: Path to wandb run directory
        
    Returns:
        DataFrame with real training metrics or None if extraction fails
    """
    run_path = Path(run_dir)
    
    print(f"Extracting real data from {run_dir}")
    
    # Find the wandb file
    wandb_files = list(run_path.glob("*.wandb"))
    if not wandb_files:
        print("No .wandb files found")
        return None
    
    wandb_file = wandb_files[0]
    print(f"Processing wandb file: {wandb_file}")
    
    # Try to parse the wandb file
    df = parse_wandb_file_native(str(wandb_file))
    
    if df is not None:
        # Clean up the data
        df = df.copy()
        
        # Ensure we have a step column
        if '_step' in df.columns:
            df['step'] = df['_step']
        elif 'step' not in df.columns:
            df['step'] = range(len(df))
        
        # Sort by step
        df = df.sort_values('step').reset_index(drop=True)
        
        # Save the extracted data
        output_file = run_path / "extracted_real_data.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved real extracted data to: {output_file}")
        
        return df
    
    print("Failed to extract real data")
    return None


def main():
    """Test the real data extraction."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python real_wandb_parser.py <run_directory>")
        return
    
    run_dir = sys.argv[1]
    df = extract_real_wandb_data(run_dir)
    
    if df is not None:
        print("\nExtracted real data preview:")
        print(df.head())
        print(f"\nData info:")
        print(df.info())
        print(f"\nData description:")
        print(df.describe())
    else:
        print("Failed to extract any real data")


if __name__ == "__main__":
    main()
