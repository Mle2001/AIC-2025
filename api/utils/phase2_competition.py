#!/usr/bin/env python3
"""
VideoRAG Phase 2 - Competition Query Engine
==========================================

Lightning-fast query engine for competition phase.
Pre-loads VideoRAG index and provides optimized query interface.

Competition Usage:
    python phase2_competition.py --index_dir ./frame_index --query "Find cooking scenes"

Author: Competition Team
"""

import os
import sys
import time
import json
import logging
import warnings
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing

# VideoRAG imports
from videorag import VideoRAG, QueryParam
from videorag._llm import openai_4o_mini_config

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

class CompetitionQueryEngine:
    """
    Optimized query engine for competition phase
    """
    
    def __init__(self, 
                 index_dir: str,
                 openai_key: str,
                 fast_mode: bool = True):
        """
        Initialize competition query engine
        
        Args:
            index_dir: Directory containing pre-built VideoRAG index
            openai_key: OpenAI API key
            fast_mode: Enable fast query optimizations
        """
        self.index_dir = Path(index_dir)
        self.fast_mode = fast_mode
        
        # Setup environment
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Setup logging
        self._setup_logging()
        
        # Load VideoRAG with pre-built index
        self.logger.info("🚀 Initializing Competition Query Engine...")
        self._load_videorag()
        
        # Load metadata
        self._load_metadata()
        
        # Setup query parameters
        self._setup_query_params()
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.query_history = []
    
    def _setup_logging(self):
        """Setup lightweight logging for competition"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger('CompetitionEngine')
    
    def _load_videorag(self):
        """Load VideoRAG with pre-built index"""
        try:
            self.logger.info(f"📁 Loading VideoRAG index from: {self.index_dir}")
            
            self.videorag = VideoRAG(
                llm=openai_4o_mini_config,
                working_dir=str(self.index_dir)
            )
            
            # Load caption model (cached from Phase 1)
            self.videorag.load_caption_model(debug=False)
            
            self.logger.info("✅ VideoRAG loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load VideoRAG: {e}")
            raise
    
    def _load_metadata(self):
        """Load processing metadata from Phase 1"""
        manifest_path = self.index_dir / 'frame_sequences_manifest.json'
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"📊 Loaded metadata: {self.metadata['total_sequences']} sequences")
        else:
            self.logger.warning("⚠️  No metadata found - using default settings")
            self.metadata = {}
    
    def _setup_query_params(self):
        """Setup optimized query parameters"""
        self.fast_param = QueryParam(mode="videorag")
        
        if self.fast_mode:
            # Fast mode optimizations
            self.fast_param.wo_reference = True
            self.fast_param.max_retrieval_results = 5
            self.fast_param.temperature = 0.1
        else:
            # Accuracy mode
            self.fast_param.wo_reference = False
            self.fast_param.max_retrieval_results = 10
            self.fast_param.temperature = 0.3
        
        self.logger.info(f"⚡ Query mode: {'FAST' if self.fast_mode else 'ACCURATE'}")
    
    def parse_response_timestamps(self, response: str, video_folder: str = None) -> Dict:
        """
        Parse VideoRAG response to extract timestamps and video information
        
        Args:
            response: VideoRAG response text
            video_folder: Optional hint for video folder name
            
        Returns:
            Dictionary with parsed results
        """
        result = {
            'video_folder': None,
            'start_time': None,
            'end_time': None,
            'confidence': 0.5,
            'raw_response': response
        }
        
        try:
            response_text = str(response).lower()
            
            # Extract video/folder references
            video_patterns = [
                r'video[_\s]*(\d+)',
                r'folder[_\s]*(\d+)',
                r'sequence[_\s]*(\d+)',
                r'clip[_\s]*(\d+)',
                r'([a-zA-Z0-9_-]+\.mp4)',
                r'([a-zA-Z0-9_-]+\.mkv)',
                r'([a-zA-Z0-9_-]+)/',
            ]
            
            for pattern in video_patterns:
                matches = re.findall(pattern, response_text)
                if matches:
                    result['video_folder'] = matches[0]
                    break
            
            # Extract timestamps - multiple formats
            time_patterns = [
                # HH:MM:SS format
                r'(\d{1,2}):(\d{2}):(\d{2})',
                # MM:SS format
                r'(\d{1,2}):(\d{2})',
                # Seconds only
                r'(\d+\.?\d*)\s*seconds?',
                r'(\d+\.?\d*)\s*s\b',
                # Frame numbers (convert to time)
                r'frame[s]?\s*(\d+)',
                # Time ranges
                r'from\s*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)',
                r'between\s*(\d+\.?\d*)\s*and\s*(\d+\.?\d*)',
                # Minute markers
                r'at\s*(\d+\.?\d*)\s*minutes?',
            ]
            
            times_found = []
            
            for pattern in time_patterns:
                matches = re.findall(pattern, response_text)
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) == 3:  # HH:MM:SS
                            h, m, s = map(float, match)
                            total_seconds = h * 3600 + m * 60 + s
                        elif len(match) == 2:  # MM:SS or range
                            try:
                                m, s = map(float, match)
                                total_seconds = m * 60 + s
                            except:
                                # Might be a range
                                total_seconds = float(match[0])
                                times_found.append(float(match[1]))
                        else:
                            total_seconds = float(match[0])
                    else:
                        total_seconds = float(match)
                    
                    times_found.append(total_seconds)
            
            # Set start and end times
            if times_found:
                times_found.sort()
                result['start_time'] = times_found[0]
                
                if len(times_found) > 1:
                    result['end_time'] = times_found[-1]
                else:
                    # Default 10-second segment
                    result['end_time'] = result['start_time'] + 10.0
                
                result['confidence'] = 0.8
            
            # Convert frame numbers to timestamps if no time found
            if not times_found:
                frame_pattern = r'frame[s]?\s*(\d+)'
                frame_matches = re.findall(frame_pattern, response_text)
                
                if frame_matches:
                    fps = self.metadata.get('sequences', [{}])[0].get('metadata', {}).get('fps', 30.0)
                    start_frame = int(frame_matches[0])
                    result['start_time'] = start_frame / fps
                    result['end_time'] = result['start_time'] + 10.0
                    result['confidence'] = 0.7
        
        except Exception as e:
            self.logger.warning(f"⚠️  Error parsing response: {e}")
        
        return result
    
    def query(self, query_text: str, timeout: float = 30.0) -> Dict:
        """
        Execute competition query with timeout
        
        Args:
            query_text: Search query text
            timeout: Query timeout in seconds
            
        Returns:
            Competition result dictionary
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Query: {query_text[:50]}...")
            
            # Execute VideoRAG query
            response = self.videorag.query(
                query=query_text, 
                param=self.fast_param
            )
            
            query_time = time.time() - start_time
            
            # Parse response
            parsed_result = self.parse_response_timestamps(response)
            
            # Build competition result
            result = {
                'query': query_text,
                'video_folder': parsed_result['video_folder'],
                'start_timestamp': parsed_result['start_time'],
                'end_timestamp': parsed_result['end_time'],
                'confidence': parsed_result['confidence'],
                'query_time': query_time,
                'status': 'success',
                'raw_response': str(response)
            }
            
            # Update statistics
            self.query_count += 1
            self.total_query_time += query_time
            self.query_history.append(result)
            
            self.logger.info(f"✅ Query completed in {query_time:.2f}s")
            
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            
            error_result = {
                'query': query_text,
                'video_folder': None,
                'start_timestamp': None,
                'end_timestamp': None,
                'confidence': 0.0,
                'query_time': query_time,
                'status': 'failed',
                'error': str(e)
            }
            
            self.logger.error(f"❌ Query failed: {e}")
            return error_result
    
    def batch_query(self, queries: List[str]) -> List[Dict]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of query strings
            
        Returns:
            List of query results
        """
        self.logger.info(f"📦 Processing {len(queries)} queries in batch...")
        
        results = []
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Query {i}/{len(queries)}")
            result = self.query(query)
            results.append(result)
        
        batch_time = time.time() - start_time
        avg_time = batch_time / len(queries)
        
        self.logger.info(f"📊 Batch completed: {batch_time:.2f}s total, {avg_time:.2f}s avg")
        
        return results
    
    def format_competition_output(self, result: Dict) -> str:
        """
        Format result for competition submission
        
        Args:
            result: Query result dictionary
            
        Returns:
            Formatted output string
        """
        if result['status'] != 'success':
            return f"ERROR: {result.get('error', 'Query failed')}"
        
        video_folder = result['video_folder'] or 'unknown'
        start_time = result['start_timestamp'] or 0.0
        end_time = result['end_timestamp'] or 10.0
        
        # Format timestamps
        def format_time(seconds):
            if seconds is None:
                return "00:00"
            
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        
        return f"Video: {video_folder}, Time: {format_time(start_time)} - {format_time(end_time)}"
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if self.query_count == 0:
            return {'message': 'No queries processed yet'}
        
        avg_time = self.total_query_time / self.query_count
        success_rate = sum(1 for q in self.query_history if q['status'] == 'success') / self.query_count
        
        return {
            'total_queries': self.query_count,
            'average_query_time': avg_time,
            'total_time': self.total_query_time,
            'success_rate': success_rate,
            'mode': 'FAST' if self.fast_mode else 'ACCURATE'
        }
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n🎯 VideoRAG Competition Query Engine")
        print("=" * 50)
        print("Enter queries (or 'quit' to exit, 'stats' for performance)")
        print()
        
        while True:
            try:
                query = input("🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    stats = self.get_performance_stats()
                    print(f"📊 Performance: {json.dumps(stats, indent=2)}")
                    continue
                elif not query:
                    continue
                
                # Execute query
                result = self.query(query)
                
                # Display result
                output = self.format_competition_output(result)
                print(f"📄 Result: {output}")
                print(f"⏱️  Time: {result['query_time']:.2f}s")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point for competition query engine"""
    parser = argparse.ArgumentParser(description='VideoRAG Competition Query Engine')
    parser.add_argument('--index_dir', required=True, help='Directory with pre-built VideoRAG index')
    parser.add_argument('--openai_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--query', help='Single query to execute')
    parser.add_argument('--queries_file', help='File containing queries (one per line)')
    parser.add_argument('--output_file', help='Output file for results')
    parser.add_argument('--fast_mode', action='store_true', default=True, help='Enable fast mode optimizations')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Get OpenAI key
    openai_key = args.openai_key or os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Error: OpenAI API key required!")
        print("Set --openai_key argument or OPENAI_API_KEY environment variable")
        return 1
    
    # Initialize engine
    try:
        engine = CompetitionQueryEngine(
            index_dir=args.index_dir,
            openai_key=openai_key,
            fast_mode=args.fast_mode
        )
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return 1
    
    # Execute based on mode
    try:
        if args.interactive:
            # Interactive mode
            engine.interactive_mode()
            
        elif args.query:
            # Single query
            result = engine.query(args.query)
            output = engine.format_competition_output(result)
            print(output)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(output + '\n')
        
        elif args.queries_file:
            # Batch queries
            if not Path(args.queries_file).exists():
                print(f"❌ Queries file not found: {args.queries_file}")
                return 1
            
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = engine.batch_query(queries)
            
            # Output results
            output_lines = []
            for result in results:
                output = engine.format_competition_output(result)
                print(output)
                output_lines.append(output)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write('\n'.join(output_lines) + '\n')
        
        else:
            print("❌ Error: Specify --query, --queries_file, or --interactive")
            return 1
        
        # Show final stats
        stats = engine.get_performance_stats()
        if stats.get('total_queries', 0) > 0:
            print(f"\n📊 Final Stats: {stats['total_queries']} queries, "
                  f"{stats['average_query_time']:.2f}s avg, "
                  f"{stats['success_rate']:.1%} success rate")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return 1


# Utility functions for competition
def quick_query(index_dir: str, query: str, openai_key: str = None) -> str:
    """Quick query function for simple usage"""
    openai_key = openai_key or os.environ.get('OPENAI_API_KEY')
    
    engine = CompetitionQueryEngine(
        index_dir=index_dir,
        openai_key=openai_key,
        fast_mode=True
    )
    
    result = engine.query(query)
    return engine.format_competition_output(result)


def competition_server(index_dir: str, openai_key: str = None, port: int = 8000):
    """Simple HTTP server for competition queries"""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        import json
    except ImportError:
        print("❌ HTTP server requires Python standard library")
        return
    
    # Initialize engine
    engine = CompetitionQueryEngine(
        index_dir=index_dir,
        openai_key=openai_key or os.environ.get('OPENAI_API_KEY'),
        fast_mode=True
    )
    
    class QueryHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/query?'):
                # Parse query parameter
                query_params = urllib.parse.parse_qs(self.path.split('?', 1)[1])
                query_text = query_params.get('q', [''])[0]
                
                if query_text:
                    # Execute query
                    result = engine.query(query_text)
                    output = engine.format_competition_output(result)
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(output.encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'Missing query parameter: ?q=your_query')
            else:
                self.send_response(404)
                self.end_headers()
    
    print(f"🌐 Starting competition server on port {port}")
    print(f"🔍 Query URL: http://localhost:{port}/query?q=your_query_here")
    
    httpd = HTTPServer(('localhost', port), QueryHandler)
    httpd.serve_forever()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    exit(main())