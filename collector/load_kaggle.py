import os
import pandas as pd
import requests
import datetime as dt
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

COLLECTOR_URL = os.environ.get("COLLECTOR_POST", "http://localhost:8100/collect/batch")

def csv_to_events(csv_path: str, limit=None):
    try:
        # Try different encodings and delimiters
        try:
            df = pd.read_csv(csv_path)
        except:
            # Try with different encoding
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Check if we have the expected columns, if not, try to detect them
        expected_columns = ['Date', 'Customer_Name', 'Product', 'Total_Items', 'Total_Cost', 
                           'Payment_Method', 'City', 'Store_Type', 'Discount_Applied',
                           'Customer_Category', 'Season', 'Promotion']
        
        available_columns = df.columns.tolist()
        print(f"Available columns in CSV: {available_columns}")
        
        # Create a mapping from expected column names to actual column names.
        column_mapping = {}
        for expected_col in expected_columns:
            # Try exact match first
            if expected_col in available_columns:
                column_mapping[expected_col] = expected_col
            else:
                # Try case-insensitive match
                for actual_col in available_columns:
                    if actual_col.lower() == expected_col.lower():
                        column_mapping[expected_col] = actual_col
                        break
                # If still not found, use the first column that might match
                if expected_col not in column_mapping:
                    for actual_col in available_columns:
                        if expected_col.lower() in actual_col.lower():
                            column_mapping[expected_col] = actual_col
                            break
        
        print(f"Column mapping: {column_mapping}")
        
        if limit:
            df = df.head(limit)

        events = []
        for idx, row in df.iterrows():
            try:
                # Use mapped column names with fallbacks
                date_col = column_mapping.get('Date', df.columns[0])
                ts = pd.to_datetime(row[date_col]).to_pydatetime() if pd.notna(row[date_col]) else dt.datetime.utcnow()
                
                # Extract store ID from City or use another column
                city_col = column_mapping.get('City', 'City')
                store_id = str(row[city_col]) if city_col in df.columns and pd.notna(row[city_col]) else f"Store{idx % 3 + 1}"
                
                # Parse products
                product_col = column_mapping.get('Product', 'Product')
                products_str = str(row[product_col]) if product_col in df.columns else "Unknown"
                
                if products_str.startswith('[') and products_str.endswith(']'):
                    try:
                        products = json.loads(products_str.replace("'", '"'))
                    except:
                        products = [p.strip().strip("'") for p in products_str.strip('[]').split(',')]
                else:
                    products = [products_str]
                
                # Get quantity and amount
                qty_col = column_mapping.get('Total_Items', 'Total_Items')
                amount_col = column_mapping.get('Total_Cost', 'Total_Cost')
                
                qty = int(row[qty_col]) if qty_col in df.columns and pd.notna(row[qty_col]) else 1
                amount = float(row[amount_col]) if amount_col in df.columns and pd.notna(row[amount_col]) else 0.0
                
                # Create enhanced payload with all your rich data
                payload = {
                    "amount": amount,
                    "items": products,
                    "qty": qty,
                }
                
                # Add optional fields if they exist
                optional_fields = {
                    'customer_name': column_mapping.get('Name'),
                    'payment_method': column_mapping.get('Payment_Method'),
                    'store_type': column_mapping.get('Store_Type'),
                    'discount_applied': column_mapping.get('Discount_Applied'),
                    'customer_category': column_mapping.get('Customer_Category'),
                    'season': column_mapping.get('Season'),
                    'promotion': column_mapping.get('Promotion')
                }
                
                for field, col_name in optional_fields.items():
                    if col_name and col_name in df.columns and pd.notna(row[col_name]):
                        if field == 'discount_applied':
                            payload[field] = bool(row[col_name])
                        else:
                            payload[field] = str(row[col_name])
                    else:
                        payload[field] = "Unknown" if field != 'discount_applied' else False

                events.append({
                    "event_id": f"tx{idx}",
                    "store_id": store_id,
                    "ts": ts.isoformat(),
                    "event_type": "sale",
                    "payload": payload
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
                
        return events
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def send_events(events, batch_size=200):
    if not events:
        print("No events to send!")
        return
        
    for i in range(0, len(events), batch_size):
        chunk = events[i:i+batch_size]
        try:
            r = requests.post(COLLECTOR_URL, json=chunk, timeout=60)
            r.raise_for_status()
            print(f"Sent {i+len(chunk)}/{len(events)} → {r.json()}")
        except Exception as e:
            print(f"Failed to send batch {i}-{i+len(chunk)}: {e}")

if __name__ == "__main__":
    # Try multiple possible paths for the CSV file
    possible_paths = [
        "data/Retail_Transactions_Dataset.csv",
        "collector/data/Retail_Transactions_Dataset.csv", 
        "../data/Retail_Transactions_Dataset.csv",
        "Retail_Transactions_Dataset.csv",
        "../Retail_Transactions_Dataset.csv"
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if not csv_path:
        print("ERROR: Could not find Retail_Transactions_Dataset.csv")
        print("Please place the CSV file in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        exit(1)
    
    # First, let's just inspect the CSV file to see what columns it actually has
    try:
        test_df = pd.read_csv(csv_path)
        print("=" * 50)
        print("CSV FILE ANALYSIS")
        print("=" * 50)
        print(f"File: {csv_path}")
        print(f"Shape: {test_df.shape}")
        print(f"Columns: {list(test_df.columns)}")
        print(f"First few rows:")
        print(test_df.head(3))
        print("=" * 50)
    except Exception as e:
        print(f"Failed to analyze CSV: {e}")
        # Try with different encoding
        try:
            test_df = pd.read_csv(csv_path, encoding='latin-1')
            print(f"Columns with latin-1 encoding: {list(test_df.columns)}")
        except:
            print("Could not read CSV with any encoding")
    
    limit = int(os.environ.get("LIMIT", "50"))  # Start with small limit for testing
    batch = int(os.environ.get("BATCH", "20"))

    print(f"Loading data from {csv_path}...")
    events = csv_to_events(csv_path, limit=limit)
    print(f"Created {len(events)} events")
    
    if events:
        print("Sample event:")
        print(json.dumps(events[0], indent=2))
        
    send_events(events, batch_size=batch)


    """ import os
import pandas as pd
import requests
import datetime as dt
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

COLLECTOR_URL = os.environ.get("COLLECTOR_POST", "http://localhost:8100/collect/batch")

def csv_to_events(csv_path: str, limit=None):
    try:
        # Try different encodings and delimiters
        try:
            df = pd.read_csv(csv_path)
        except:
            # Try with different encoding
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Check if we have the expected columns, if not, try to detect them
        expected_columns = ['Date', 'Name', 'Product', 'Total_Items', 'Total_Cost', 
                           'Payment_Method', 'City', 'Store_Type', 'Discount_Applied',
                           'Customer_Category', 'Season', 'Promotion']
        
        available_columns = df.columns.tolist()
        print(f"Available columns in CSV: {available_columns}")
        
        # Create a mapping from expected column names to actual column names
        column_mapping = {}
        for expected_col in expected_columns:
            # Try exact match first
            if expected_col in available_columns:
                column_mapping[expected_col] = expected_col
            else:
                # Try case-insensitive match
                for actual_col in available_columns:
                    if actual_col.lower() == expected_col.lower():
                        column_mapping[expected_col] = actual_col
                        break
                # If still not found, use the first column that might match
                if expected_col not in column_mapping:
                    for actual_col in available_columns:
                        if expected_col.lower() in actual_col.lower():
                            column_mapping[expected_col] = actual_col
                            break
        
        print(f"Column mapping: {column_mapping}")
        
        if limit:
            df = df.head(limit)

        events = []
        for idx, row in df.iterrows():
            try:
                # Use mapped column names with fallbacks
                date_col = column_mapping.get('Date', df.columns[0])
                ts = pd.to_datetime(row[date_col]).to_pydatetime() if pd.notna(row[date_col]) else dt.datetime.utcnow()
                
                # Extract store ID from City or use another column
                city_col = column_mapping.get('City', 'City')
                store_id = str(row[city_col]) if city_col in df.columns and pd.notna(row[city_col]) else f"Store{idx % 3 + 1}"
                
                # Parse products
                product_col = column_mapping.get('Product', 'Product')
                products_str = str(row[product_col]) if product_col in df.columns else "Unknown"
                
                if products_str.startswith('[') and products_str.endswith(']'):
                    try:
                        products = json.loads(products_str.replace("'", '"'))
                    except:
                        products = [p.strip().strip("'") for p in products_str.strip('[]').split(',')]
                else:
                    products = [products_str]
                
                # Get quantity and amount
                qty_col = column_mapping.get('Total_Items', 'Total_Items')
                amount_col = column_mapping.get('Total_Cost', 'Total_Cost')
                
                qty = int(row[qty_col]) if qty_col in df.columns and pd.notna(row[qty_col]) else 1
                amount = float(row[amount_col]) if amount_col in df.columns and pd.notna(row[amount_col]) else 0.0
                
                # Create enhanced payload with all your rich data
                payload = {
                    "amount": amount,
                    "items": products,
                    "qty": qty,
                }
                
                # Add optional fields if they exist
                optional_fields = {
                    'customer_name': column_mapping.get('Name'),
                    'payment_method': column_mapping.get('Payment_Method'),
                    'store_type': column_mapping.get('Store_Type'),
                    'discount_applied': column_mapping.get('Discount_Applied'),
                    'customer_category': column_mapping.get('Customer_Category'),
                    'season': column_mapping.get('Season'),
                    'promotion': column_mapping.get('Promotion')
                }
                
                for field, col_name in optional_fields.items():
                    if col_name and col_name in df.columns and pd.notna(row[col_name]):
                        if field == 'discount_applied':
                            payload[field] = bool(row[col_name])
                        else:
                            payload[field] = str(row[col_name])
                    else:
                        payload[field] = "Unknown" if field != 'discount_applied' else False

                events.append({
                    "event_id": f"tx{idx}",
                    "store_id": store_id,
                    "ts": ts.isoformat(),
                    "event_type": "sale",
                    "payload": payload
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
                
        return events
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def send_events(events, batch_size=200):
    if not events:
        print("No events to send!")
        return
        
    for i in range(0, len(events), batch_size):
        chunk = events[i:i+batch_size]
        try:
            r = requests.post(COLLECTOR_URL, json=chunk, timeout=60)
            r.raise_for_status()
            print(f"Sent {i+len(chunk)}/{len(events)} → {r.json()}")
        except Exception as e:
            print(f"Failed to send batch {i}-{i+len(chunk)}: {e}")

if __name__ == "__main__":
    # Try multiple possible paths for the CSV file
    possible_paths = [
        "data/Retail_Transactions_Dataset.csv",
        "collector/data/Retail_Transactions_Dataset.csv", 
        "../data/Retail_Transactions_Dataset.csv",
        "Retail_Transactions_Dataset.csv",
        "../Retail_Transactions_Dataset.csv"
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if not csv_path:
        print("ERROR: Could not find Retail_Transactions_Dataset.csv")
        print("Please place the CSV file in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        exit(1)
    
    # First, let's just inspect the CSV file to see what columns it actually has
    try:
        test_df = pd.read_csv(csv_path)
        print("=" * 50)
        print("CSV FILE ANALYSIS")
        print("=" * 50)
        print(f"File: {csv_path}")
        print(f"Shape: {test_df.shape}")
        print(f"Columns: {list(test_df.columns)}")
        print(f"First few rows:")
        print(test_df.head(3))
        print("=" * 50)
    except Exception as e:
        print(f"Failed to analyze CSV: {e}")
        # Try with different encoding
        try:
            test_df = pd.read_csv(csv_path, encoding='latin-1')
            print(f"Columns with latin-1 encoding: {list(test_df.columns)}")
        except:
            print("Could not read CSV with any encoding")
    
    limit = int(os.environ.get("LIMIT", "50"))  # Start with small limit for testing
    batch = int(os.environ.get("BATCH", "20"))

    print(f"Loading data from {csv_path}...")
    events = csv_to_events(csv_path, limit=limit)
    print(f"Created {len(events)} events")
    
    if events:
        print("Sample event:")
        print(json.dumps(events[0], indent=2))
        
    send_events(events, batch_size=batch)"""