# services/folder_scanner.py - Enhanced with batch processing and progress
from flask import Flask, request, jsonify
import os
import hashlib
import uuid
from datetime import datetime
import requests
import threading
import time

app = Flask(__name__)

class OptimizedFolderScanner:
    def __init__(self, base_path=r"E:\Innova_Intern\DEMO\utils\demo_cicd_logs", db_service_url="http://localhost:5001"):
        self.base_path = base_path
        self.db_service_url = db_service_url
        self.batch_size = 10  # Process files in batches
        print(f"🔍 Optimized Folder Scanner initialized - Base path: {self.base_path}")
    
    def scan_folders_with_progress(self):
        """Scan folders with progress tracking and batch processing"""
        print(f"🔍 Starting optimized folder scan at {datetime.now()}")
        
        if not os.path.exists(self.base_path):
            print(f"❌ Base path does not exist: {self.base_path}")
            return []
        
        # First pass: collect all files
        all_files = []
        total_files_found = 0
        
        print("📊 Phase 1: Discovering all log files...")
        for root, dirs, files in os.walk(self.base_path):
            log_files = [f for f in files if f.endswith('.log')]
            if log_files:
                print(f"📂 Found {len(log_files)} log files in: {root}")
            
            for file in log_files:
                total_files_found += 1
                file_path = os.path.join(root, file)
                
                # Extract metadata from path
                path_parts = root.replace(self.base_path, '').strip(os.sep).split(os.sep)
                if len(path_parts) >= 4:
                    tool, project, environment, server = path_parts[:4]
                    
                    log_type = file.split('_')[0]
                    status = self.extract_status_from_filename(file)
                    
                    file_info = {
                        "file_path": file_path,
                        "tool": tool,
                        "project": project,
                        "environment": environment,
                        "server": server,
                        "log_type": log_type,
                        "status": status,
                        "correlation_id": str(uuid.uuid4())[:8]
                    }
                    all_files.append(file_info)
        
        print(f"📊 Discovery complete: {total_files_found} log files found")
        
        # Second pass: process in batches
        print("📊 Phase 2: Processing files in batches...")
        new_files = []
        
        for i in range(0, len(all_files), self.batch_size):
            batch = all_files[i:i + self.batch_size]
            print(f"🔄 Processing batch {i//self.batch_size + 1}/{(len(all_files)-1)//self.batch_size + 1} ({len(batch)} files)")
            
            batch_new_files = self.process_file_batch(batch)
            new_files.extend(batch_new_files)
            
            # Progress update
            progress = ((i + len(batch)) / len(all_files)) * 100
            print(f"📈 Progress: {progress:.1f}% ({i + len(batch)}/{len(all_files)} files checked)")
        
        print(f"📊 Scan summary:")
        print(f"   Total log files found: {total_files_found}")
        print(f"   New files to process: {len(new_files)}")
        
        return new_files
    
    def process_file_batch(self, file_batch):
        """Process a batch of files"""
        new_files = []
        
        for file_info in file_batch:
            # Calculate checksum
            checksum = self.calculate_file_checksum(file_info['file_path'])
            if not checksum:
                continue
            
            # Check if file already exists in DB
            if not self.file_exists_in_db(checksum):
                file_info['checksum'] = checksum
                new_files.append(file_info)
                print(f"📄 New: {file_info['tool']}/{file_info['log_type']} ({file_info['status']})")
            else:
                print(f"⏭️  Exists: {os.path.basename(file_info['file_path'])}")
        
        return new_files
    
    def store_files_in_batches(self, files):
        """Store files in database using batch processing"""
        if not files:
            print("ℹ️  No files to store")
            return 0
        
        print(f"💾 Storing {len(files)} files in database using batch processing...")
        stored_count = 0
        failed_count = 0
        
        # Process in batches for better performance
        for i in range(0, len(files), self.batch_size):
            batch = files[i:i + self.batch_size]
            print(f"📤 Storing batch {i//self.batch_size + 1}/{(len(files)-1)//self.batch_size + 1} ({len(batch)} files)")
            
            batch_stored, batch_failed = self.store_file_batch(batch)
            stored_count += batch_stored
            failed_count += batch_failed
            
            # Progress update
            progress = ((i + len(batch)) / len(files)) * 100
            print(f"💾 Storage progress: {progress:.1f}% ({stored_count} stored, {failed_count} failed)")
        
        print(f"📊 Storage summary:")
        print(f"   Successfully stored: {stored_count}")
        print(f"   Failed to store: {failed_count}")
        
        return stored_count
    
    def store_file_batch(self, file_batch):
        """Store a batch of files"""
        stored_count = 0
        failed_count = 0
        
        for file_info in file_batch:
            try:
                # Read file content
                with open(file_info['file_path'], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Prepare data for DB service
                data = {
                    **file_info,
                    "log_content": content,
                    "file_size": len(content),
                    "processed": False
                }
                
                # Store via DB service
                response = requests.post(
                    f"{self.db_service_url}/store-log", 
                    json=data, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    stored_count += 1
                else:
                    failed_count += 1
                    print(f"   ❌ Storage failed: {response.status_code}")
                
            except Exception as e:
                failed_count += 1
                print(f"❌ Error processing {file_info['file_path']}: {e}")
        
        return stored_count, failed_count
    
    # ... (keep existing helper methods)
    def calculate_file_checksum(self, file_path):
        """Calculate MD5 checksum for file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            return None
    
    def extract_status_from_filename(self, filename):
        """Extract status (error/success) from filename"""
        if "_error_" in filename:
            return "error"
        elif "_success_" in filename:
            return "success"
        else:
            return "unknown"
    
    def file_exists_in_db(self, checksum):
        """Check if file exists in database"""
        try:
            response = requests.get(f"{self.db_service_url}/check-file/{checksum}", timeout=10)
            return response.status_code == 200 and response.json().get('exists', False)
        except:
            return False

scanner_service = OptimizedFolderScanner()

@app.route('/scan', methods=['POST'])
def scan_folders():
    """Optimized scan endpoint with batch processing"""
    try:
        print("\n" + "="*50)
        print("🔍 OPTIMIZED FOLDER SCAN REQUEST RECEIVED")
        print("="*50)
        
        # Use optimized scanning
        new_files = scanner_service.scan_folders_with_progress()
        stored_count = scanner_service.store_files_in_batches(new_files)
        
        result = {
            'status': 'success',
            'files_found': len(new_files),
            'files_stored': stored_count,
            'timestamp': datetime.utcnow().isoformat(),
            'scan_path': scanner_service.base_path,
            'processing_method': 'batch_optimized'
        }
        
        print(f"✅ OPTIMIZED SCAN COMPLETED: {result}")
        print("="*50)
        
        return jsonify(result), 200
        
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"❌ SCAN FAILED: {error_result}")
        return jsonify(error_result), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check with path verification"""
    path_exists = os.path.exists(scanner_service.base_path)
    return jsonify({
        'status': 'healthy', 
        'service': 'optimized_folder_scanner',
        'base_path': scanner_service.base_path,
        'path_exists': path_exists,
        'batch_size': scanner_service.batch_size
    }), 200

if __name__ == '__main__':
    print("🔍 Starting Optimized Folder Scanner Service")
    print(f"📁 Base path: {os.path.abspath(scanner_service.base_path)}")
    print(f"📦 Batch size: {scanner_service.batch_size}")
    app.run(debug=False, port=5002, use_reloader=False, host='0.0.0.0')
