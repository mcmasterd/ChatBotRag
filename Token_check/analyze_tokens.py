#!/usr/bin/env python3
# /home/skynet1/ChatBotRag/venv/bin/python /home/skynet1/ChatBotRag/Token_check/analyze_tokens.py
"""
📊 Token Usage Analyzer - Phân tích log token dễ đọc
"""
import csv
import os
from datetime import datetime
from collections import defaultdict

def format_cost(cost):
    """Format cost với màu sắc"""
    cost_float = float(cost)
    if cost_float > 0.001:
        return f"🔴 ${cost:.6f}"  # Đỏ cho chi phí cao
    elif cost_float > 0.0001:
        return f"🟡 ${cost:.6f}"  # Vàng cho chi phí trung bình
    else:
        return f"🟢 ${cost:.6f}"  # Xanh cho chi phí thấp

def format_tokens(total_tokens):
    """Format tokens với biểu tượng"""
    if total_tokens > 2000:
        return f"🔥 {total_tokens:,} tokens"  # Lửa cho tokens cao
    elif total_tokens > 500:
        return f"⚡ {total_tokens:,} tokens"  # Sét cho tokens trung bình
    else:
        return f"✅ {total_tokens:,} tokens"  # Check cho tokens thấp

def format_operation(op_type):
    """Format operation với biểu tượng"""
    icons = {
        'category_classification': '🔍',
        'embedding_generation': '🧮', 
        'vector_search_results': '📚',
        'session_history': '💭',
        'final_prompt': '📝',
        'main_response': '🤖',
        'main_streaming': '🌊',
        'small_talk_response': '💬',
        'cache_hit': '⚡'
    }
    icon = icons.get(op_type, '❓')
    return f"{icon} {op_type}"

def analyze_token_log(log_file):
    """Phân tích token log với format đẹp"""
    if not os.path.exists(log_file):
        print("❌ Không tìm thấy file log!")
        return
    
    print("🎯 TOKEN USAGE ANALYZER")
    print("=" * 60)
    
    # Đọc và phân tích data
    operations = []
    total_cost = 0
    total_tokens = 0
    model_stats = defaultdict(lambda: {'tokens': 0, 'cost': 0, 'count': 0})
    operation_stats = defaultdict(lambda: {'tokens': 0, 'cost': 0, 'count': 0})
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            operations.append(row)
            tokens = int(row['Total_Tokens'])
            cost = float(row['Cost_USD'])
            model = row['Model']
            op_type = row['Operation_Type']
            
            total_tokens += tokens
            total_cost += cost
            
            model_stats[model]['tokens'] += tokens
            model_stats[model]['cost'] += cost
            model_stats[model]['count'] += 1
            
            operation_stats[op_type]['tokens'] += tokens
            operation_stats[op_type]['cost'] += cost
            operation_stats[op_type]['count'] += 1
    
    # 1. Tổng quan
    print(f"📈 TỔNG QUAN")
    print(f"   💰 Tổng chi phí: {format_cost(total_cost)}")
    print(f"   🔢 Tổng tokens: {format_tokens(total_tokens)}")
    print(f"   📊 Số operations: {len(operations)}")
    print()
    
    # 2. Top 5 operations gần nhất
    print(f"🕐 TOP 5 OPERATIONS GẦN NHẤT")
    print("-" * 60)
    recent_ops = operations[-5:]
    for i, op in enumerate(reversed(recent_ops), 1):
        time = op['Timestamp']
        user = op['User_ID'][:15] + "..." if len(op['User_ID']) > 15 else op['User_ID']
        operation = format_operation(op['Operation_Type'])
        model = op['Model']
        tokens = format_tokens(int(op['Total_Tokens']))
        cost = format_cost(float(op['Cost_USD']))
        processing_time = float(op['Processing_Time_Seconds'])
        
        print(f"{i}. [{time}] {user}")
        print(f"   {operation} | {model}")
        print(f"   {tokens} | {cost} | ⏱️ {processing_time:.2f}s")
        if op['Input_Text_Preview']:
            preview = op['Input_Text_Preview'][:50] + "..." if len(op['Input_Text_Preview']) > 50 else op['Input_Text_Preview']
            print(f"   💭 Input: {preview}")
        print()
    
    # 3. Thống kê theo Model
    print(f"🔧 THỐNG KÊ THEO MODEL")
    print("-" * 60)
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['tokens'], reverse=True):
        print(f"🤖 {model}")
        print(f"   {format_tokens(stats['tokens'])} | {format_cost(stats['cost'])} | {stats['count']} calls")
        avg_tokens = stats['tokens'] / stats['count'] if stats['count'] > 0 else 0
        print(f"   📊 Avg: {avg_tokens:.0f} tokens/call")
        print()
    
    # 4. Thống kê theo Operation
    print(f"⚙️ THỐNG KÊ THEO OPERATION")
    print("-" * 60)
    for op_type, stats in sorted(operation_stats.items(), key=lambda x: x[1]['tokens'], reverse=True):
        print(f"{format_operation(op_type)}")
        print(f"   {format_tokens(stats['tokens'])} | {format_cost(stats['cost'])} | {stats['count']} calls")
        avg_tokens = stats['tokens'] / stats['count'] if stats['count'] > 0 else 0
        print(f"   📊 Avg: {avg_tokens:.0f} tokens/call")
        print()
    
    # 5. Phân tích requests cao tokens
    print(f"🚨 REQUESTS CAO TOKENS (>1000)")
    print("-" * 60)
    high_token_ops = [op for op in operations if int(op['Total_Tokens']) > 1000]
    if high_token_ops:
        for op in high_token_ops[-3:]:  # 3 requests gần nhất
            tokens = int(op['Total_Tokens'])
            input_tokens = int(op['Input_Tokens'])
            output_tokens = int(op['Output_Tokens'])
            
            print(f"⚠️ [{op['Timestamp']}] {op['User_ID'][:20]}...")
            print(f"   {format_operation(op['Operation_Type'])} | {op['Model']}")
            print(f"   📥 Input: {input_tokens:,} tokens")
            print(f"   📤 Output: {output_tokens:,} tokens")
            print(f"   💰 Cost: {format_cost(float(op['Cost_USD']))}")
            if op['Additional_Info']:
                print(f"   ℹ️ Info: {op['Additional_Info']}")
            print()
    else:
        print("✅ Không có requests nào >1000 tokens")
    
    # 6. Recommendations
    print(f"💡 KHUYẾN NGHỊ")
    print("-" * 60)
    
    # Check high token operations
    avg_tokens_per_request = total_tokens / len(operations) if operations else 0
    if avg_tokens_per_request > 1000:
        print("🔴 Token usage cao! Cần optimize:")
        print("   • Rút gọn system prompts")
        print("   • Giảm số lượng context từ vector search")
        print("   • Tăng cache hit rate")
        print()
    
    # Check expensive models
    gpt4_usage = model_stats.get('gpt-4o', {}).get('tokens', 0)
    total_gpt_tokens = model_stats.get('gpt-4o-mini', {}).get('tokens', 0) + gpt4_usage
    if gpt4_usage > total_gpt_tokens * 0.3:
        print("🟡 GPT-4o usage cao, cân nhắc:")
        print("   • Chỉ dùng GPT-4o cho classification")
        print("   • Dùng GPT-4o-mini cho response chính")
        print()
    
    # Check cache efficiency
    cache_hits = operation_stats.get('cache_hit', {}).get('count', 0)
    total_requests = len([op for op in operations if 'response' in op['Operation_Type']])
    if total_requests > 0:
        cache_rate = cache_hits / total_requests * 100
        if cache_rate < 10:
            print("🟡 Cache hit rate thấp, cải thiện:")
            print("   • Tăng TTL cache")
            print("   • Normalize queries tốt hơn")
            print()
    
    print("✅ Phân tích hoàn tất!")

if __name__ == "__main__":
    log_file = "/home/skynet1/ChatBotRag/logs/token_usage_log.csv"
    analyze_token_log(log_file)
