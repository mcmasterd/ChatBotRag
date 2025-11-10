#!/usr/bin/env python3
# /home/skynet1/ChatBotRag/venv/bin/python /home/skynet1/ChatBotRag/Token_check/monitor_tokens.py
"""
🔍 Real-time Token Monitor - Theo dõi token usage trực tiếp
"""
import time
import os
import csv
from datetime import datetime

def format_tokens_simple(tokens):
    if tokens > 1000:
        return f"🔥  {tokens:,}"
    elif tokens > 500:
        return f"⚡  {tokens:,}"
    else:
        return f"✅  {tokens:,}"

def format_cost_simple(cost):
    cost_float = float(cost)
    if cost_float > 0.001:
        return f"🟥  ${cost:.4f}"
    elif cost_float > 0.0001:
        return f"🟨  ${cost:.4f}"
    else:
        return f"🟩  ${cost:.4f}"

def monitor_tokens(log_file, interval=3):
    """Monitor token usage real-time"""
    print("🔍 REAL-TIME TOKEN MONITOR")
    print("=" * 50)
    print("Press Ctrl+C to stop")
    print()
    
    last_lines = 0
    
    while True:
        try:
            if not os.path.exists(log_file):
                print("⏳ Waiting for log file...")
                time.sleep(interval)
                continue
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            current_lines = len(lines)
            
            # Chỉ hiển thị lines mới
            if current_lines > last_lines:
                new_lines = lines[last_lines:current_lines]
                
                for line in new_lines:
                    if line.startswith('Timestamp'):  # Skip header
                        continue
                        
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 11:
                            timestamp = parts[0]
                            user_id = parts[1][:15] + "..." if len(parts[1]) > 15 else parts[1]
                            operation = parts[2]
                            model = parts[3]
                            total_tokens = int(parts[6])
                            cost = float(parts[9])
                            processing_time = float(parts[10])
                            
                            # Icon cho operation
                            op_icons = {
                                'category_classification': '🔍',
                                'embedding_generation': '🧮',
                                'vector_search_results': '📚',
                                'main_response': '🤖',
                                'main_streaming': '🌊',
                                'cache_hit': '⚡'
                            }
                            icon = op_icons.get(operation, '❓')
                            
                            print(f"[{timestamp}] {user_id}")
                            print(f"  {icon} {operation} | {model}")
                            print(f"  {format_tokens_simple(total_tokens)} | {format_cost_simple(cost)} | ⏱️   {processing_time:.2f}s")
                            print("\n" + "-" * 50)
                            print()
                            
                    except (ValueError, IndexError) as e:
                        continue
                        
                last_lines = current_lines
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n👋 Stopped monitoring")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    log_file = "/home/skynet1/ChatBotRag/logs/token_usage_log.csv"
    monitor_tokens(log_file)
