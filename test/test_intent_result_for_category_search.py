import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from chatbot import classify_categories_llm, get_relevant_content, collection, BM25, process_user_query
import time
import json

def detailed_get_relevant_content_new_flow(query: str, categories: list = None, final_results: int = 4):
    """Enhanced version matching the NEW FLOW in updated chatbot.py"""
    print(f"\n{'='*60}")
    print(f"🔍 PHÂN TÍCH CHI TIẾT QUÁ TRÌNH RETRIEVAL (FLOW MỚI)")
    print(f"{'='*60}")
    print(f"📝 Query: {query}")
    print(f"🏷️  Categories filter: {categories}")
    
    start = time.time()
    
    try:
        # NEW FLOW: Category filtering FIRST (matching chatbot.py logic)
        if categories:
            where_clause = {"category": {"$in": categories}}
            initial_results = 20  # Matching chatbot.py
            
            print(f"\n🎯 BƯỚC 1: Tìm kiếm semantic TRONG categories {categories}")
            print(f"   📊 Sử dụng ChromaDB where clause: {where_clause}")
            print(f"   📈 Số lượng kết quả tối đa: {initial_results}")
            print("-" * 50)
            
            results = collection.query(
                query_texts=[query],
                n_results=initial_results,
                where=where_clause,  # Filter FIRST
                include=["metadatas", "documents"]
            )
        else:
            print(f"\n🎯 BƯỚC 1: Tìm kiếm semantic TOÀN BỘ database")
            print("-" * 50)
            
            initial_results = 10
            results = collection.query(
                query_texts=[query],
                n_results=initial_results,
                include=["metadatas", "documents"]
            )
        
        print(f"✅ Tìm thấy {len(results['documents'][0]) if results['documents'] else 0} kết quả từ ChromaDB")
        
        # Show results (all should match category if filtered)
        if results['documents'] and results['documents'][0]:
            print(f"\n📋 Kết quả semantic search:")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                category = meta.get('category', 'N/A')
                doc_id = meta.get('doc_id', 'N/A')
                source = meta.get('source', 'N/A')
                preview = doc[:150].replace('\n', ' ') + "..." if len(doc) > 150 else doc
                
                # All should be ✅ if category filtered properly
                match_indicator = "✅" if not categories or category in categories else "❌"
                
                # Highlight if contains target keywords
                target_keywords = ["ưewvwevweve"]
                has_keywords = any(keyword.lower() in doc.lower() for keyword in target_keywords)
                keyword_indicator = "🎯" if has_keywords else "  "
                
                print(f"  {i+1:2d}. {match_indicator} {keyword_indicator} [{category}] {doc_id} - {source}")
                print(f"      📄 {preview}")
                print()
        
        # Prepare candidates (no additional filtering needed if where clause used)
        candidate_content = []
        
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                document_text = results['documents'][0][i]
                candidate_content.append({
                    'document': document_text,
                    'metadata': metadata,
                    'relevance_score': 0.0,
                    'original_rank': i + 1
                })
        
        print(f"\n✅ Candidates cho BM25 reranking: {len(candidate_content)}")
        
        # BM25 Reranking (matching chatbot.py logic)
        if candidate_content:
            print(f"\n🔄 BƯỚC 2: BM25 Reranking")
            print("-" * 50)
            
            documents = [item['document'] for item in candidate_content]
            bm25 = BM25(documents)
            bm25_scores = bm25.get_scores(query)
            
            print(f"📊 BM25 Scores:")
            for i, (score, item) in enumerate(zip(bm25_scores, candidate_content)):
                meta = item['metadata']
                doc_id = meta.get('doc_id', 'N/A')
                source = meta.get('source', 'N/A')
                
                # Check for target content
                has_target_content = any(keyword in item['document'].lower() 
                                       for keyword in ["10 ngày", "phản ánh", "thắc mắc"])
                target_flag = "🎯" if has_target_content else "  "
                
                print(f"  {i+1}. Score: {score:.4f} {target_flag} | {doc_id} - {source}")
            
            # Apply scores and sort
            for i, score in enumerate(bm25_scores):
                candidate_content[i]['relevance_score'] = score
                
            candidate_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            print(f"\n🏆 Thứ tự sau khi BM25 rerank:")
            for i, item in enumerate(candidate_content):
                meta = item['metadata']
                original_rank = item['original_rank']
                score = item['relevance_score']
                doc_id = meta.get('doc_id', 'N/A')
                source = meta.get('source', 'N/A')
                
                # Check target content
                has_target_content = "10 ngày" in item['document']
                target_flag = "🔥" if has_target_content else "  "
                
                # Ranking change
                rank_change = ""
                if i + 1 < original_rank:
                    rank_change = f"📈 (tăng {original_rank - (i + 1)} bậc)"
                elif i + 1 > original_rank:
                    rank_change = f"📉 (giảm {(i + 1) - original_rank} bậc)"
                else:
                    rank_change = "➡️ (giữ nguyên)"
                    
                print(f"  {i+1}. Score: {score:.4f} {target_flag} | {doc_id} - {source} {rank_change}")
            
            # Final results
            final_content = candidate_content[:final_results]
            
            print(f"\n🎯 BƯỚC 3: Kết quả cuối cùng (top {final_results})")
            print("-" * 50)
            
            found_target = False
            for i, item in enumerate(final_content):
                meta = item['metadata']
                score = item['relevance_score']
                doc_id = meta.get('doc_id', 'N/A')
                source = meta.get('source', 'N/A')
                
                # Check for target content
                if "10 ngày" in item['document']:
                    found_target = True
                    
                preview = item['document'][:250].replace('\n', ' ') + "..." if len(item['document']) > 250 else item['document']
                
                print(f"🥇 Kết quả {i+1}:")
                print(f"   📊 BM25 Score: {score:.4f}")
                print(f"   📁 Doc ID: {doc_id}")
                print(f"   📄 Source: {source}")
                
                if "10 ngày" in item['document']:
                    print(f"   🔥 *** CHỨA THÔNG TIN '10 NGÀY' ***")
                    
                print(f"   📝 Nội dung: {preview}")
                print()
            
            retrieval_time = time.time() - start
            print(f"⏱️  Thời gian xử lý: {retrieval_time:.4f}s")
            print(f"📈 Tổng kết: {len(results['documents'][0]) if results['documents'] else 0} candidates → {len(final_content)} final results")
            
            if found_target:
                print(f"🎉 SUCCESS: Tìm thấy thông tin mục tiêu trong kết quả cuối cùng!")
            else:
                print(f"⚠️  WARNING: Không tìm thấy thông tin mục tiêu trong kết quả cuối cùng")
            
            return final_content
        else:
            print("❌ Không có kết quả nào")
            return []
            
    except Exception as e:
        print(f"❌ Lỗi trong quá trình retrieval: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def test_full_pipeline(query: str, user_id: str = "test_user"):
    """Test the complete pipeline including LLM response"""
    print(f"\n🔗 KIỂM TRA TOÀN BỘ PIPELINE")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Test full chatbot response
        response = process_user_query(query, user_id)
        
        total_time = time.time() - start_time
        
        print(f"\n📝 CÂU HỎI: {query}")
        print(f"🤖 PHẢN HỒI: {response}")
        print(f"⏱️  Tổng thời gian: {total_time:.4f}s")
        
        # Quality checks
        quality_checks = []
        
        # Check 1: Response length
        if len(response) > 50:
            quality_checks.append("✅ Độ dài phản hồi hợp lý")
        else:
            quality_checks.append("⚠️  Phản hồi quá ngắn")
            
        # Check 2: Contains relevant keywords
        query_keywords = query.lower().split()
        response_lower = response.lower()
        relevant_keywords = sum(1 for word in query_keywords if word in response_lower)
        
        if relevant_keywords >= len(query_keywords) * 0.3:
            quality_checks.append("✅ Chứa từ khóa liên quan")
        else:
            quality_checks.append("⚠️  Thiếu từ khóa liên quan")
            
        # Check 3: No error messages
        error_indicators = ["lỗi", "không thể", "xin lỗi", "error"]
        has_error = any(indicator in response_lower for indicator in error_indicators)
        
        if not has_error:
            quality_checks.append("✅ Không có thông báo lỗi")
        else:
            quality_checks.append("❌ Có thông báo lỗi")
            
        # Check 4: Specific content for time-related queries
        if "thời gian" in query.lower() or "bao lâu" in query.lower():
            if "10 ngày" in response or "ngày" in response:
                quality_checks.append("✅ Chứa thông tin thời gian cụ thể")
            else:
                quality_checks.append("❌ Thiếu thông tin thời gian cụ thể")
        
        print(f"\n📊 ĐÁNH GIÁ CHẤT LƯỢNG:")
        for check in quality_checks:
            print(f"   {check}")
            
        return response
        
    except Exception as e:
        print(f"❌ Lỗi trong pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_test_suite():
    """Run a comprehensive test suite"""
    test_cases = [
        {
            "name": "Test thời gian phản ánh thắc mắc học bổng",
            "query": "Thời gian sinh viên có thể phản ánh, thắc mắc về danh sách học bổng là bao lâu?",
            "expected_keywords": ["10 ngày", "phản ánh", "thắc mắc"]
        },
        {
            "name": "Test học bổng cơ bản",
            "query": "Điều kiện để được nhận học bổng khuyến khích học tập?",
            "expected_keywords": ["học bổng", "điều kiện", "khuyến khích"]
        },
        {
            "name": "Test small talk",
            "query": "Xin chào, bạn có khỏe không?",
            "expected_keywords": ["chào", "khỏe"]
        }
    ]
    
    print(f"\n🧪 CHẠY BỘ TEST TỔNG HỢP")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        # Test category classification
        categories = classify_categories_llm(test_case['query'])
        print(f"🏷️  Categories: {categories}")
        
        # Test retrieval
        retrieval_results = get_relevant_content(test_case['query'], categories=categories)
        print(f"🔍 Retrieval results: {len(retrieval_results)}")
        
        # Test full pipeline
        response = test_full_pipeline(test_case['query'], f"test_user_{i}")
        
        # Check expected keywords
        found_keywords = []
        if response:
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    found_keywords.append(keyword)
        
        result = {
            'test_name': test_case['name'],
            'query': test_case['query'],
            'categories': categories,
            'retrieval_count': len(retrieval_results),
            'response_length': len(response) if response else 0,
            'found_keywords': found_keywords,
            'expected_keywords': test_case['expected_keywords'],
            'success': len(found_keywords) >= len(test_case['expected_keywords']) * 0.5
        }
        
        results.append(result)
        
        print(f"📊 Kết quả: {'✅ PASS' if result['success'] else '❌ FAIL'}")
        print(f"   Keywords found: {found_keywords}")
        print(f"   Expected: {test_case['expected_keywords']}")
    
    # Summary
    print(f"\n📈 TỔNG KẾT BỘ TEST")
    print("=" * 40)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {result['test_name']}")
    
    return results

def show_classification_result(result):
    if isinstance(result, dict):
        print(f"🏷️  Categories: {result.get('categories', [])}")
        if result.get('rephrased_query'):
            print(f"🔄 Đã rephrase: {result['rephrased_query']}")
        else:
            print(f"ℹ️  Không thực hiện rephrase, dùng câu hỏi gốc.")
    else:
        print(f"🏷️  Categories: {result}")

def show_search_results(results, top_n=4):
    print(f"\n🔍 Kết quả retrieval: {len(results)} documents")
    for i, item in enumerate(results[:top_n], 1):
        meta = item['metadata']
        preview = item['document'][:100].replace('\n', ' ') + "..."
        print(f"   {i}. [{meta.get('category', 'N/A')}] {meta.get('doc_id', 'N/A')}")
        print(f"      {preview}")

if __name__ == "__main__":
    print("🤖 === KIỂM TRA NHẬN DIỆN Ý ĐỊNH VÀ TÌM KIẾM (PHIÊN BẢN MỚI) ===")
    print("💡 Hệ thống đã được cập nhật để phù hợp với logic mới của chatbot.py")
    print()
    
    while True:
        print("\n🎛️  MENU CHÍNH:")
        print("1. Test intent (intent + rephrase + search)")
        print("2. Test retrieval chi tiết từng bước (giống chatbot.py)")
        print("3. Thoát")
        
        choice = input("\n[-] Chọn option (1-3): ").strip()
        
        if choice == '1':
            query = input("[-] Nhập câu hỏi: ").strip()
            if not query:
                continue
            # Step 1: Classify categories + rephrase
            print(f"\n🎯 PHÂN LOẠI CATEGORY")
            print("=" * 30)
            result = classify_categories_llm(query)
            show_classification_result(result)
            categories = result.get('categories', []) if isinstance(result, dict) else result
            rephrased_query = result.get('rephrased_query') if isinstance(result, dict) else None
            # Step 2: Search
            if categories == ['small_talk']:
                print("💬 Small talk: Bỏ qua vector search, chỉ dùng LLM trả lời.")
            else:
                vector_query = rephrased_query if rephrased_query else query
                results = get_relevant_content(vector_query, categories=categories, final_results=4)
                show_search_results(results)
        
        elif choice == '2':
            query = input("[-] Nhập câu hỏi để test chi tiết: ").strip()
            if not query:
                continue
            start = time.time()  # Đo thời gian xử lý toàn bộ bước
            # Step 1: Lấy lịch sử hội thoại (giả lập, vì test offline)
            print(f"\n🕑 Lấy lịch sử hội thoại gần nhất (giả lập)")
            session_history = []  # Có thể mở rộng lấy từ file/log nếu cần
            print(f"   Lịch sử: {session_history}")
            # Step 2: Phân loại intent + rephrase
            print(f"\n🎯 PHÂN LOẠI CATEGORY + REPHRASE")
            classify_result = classify_categories_llm(query, history=session_history)
            show_classification_result(classify_result)
            categories = classify_result.get('categories', []) if isinstance(classify_result, dict) else classify_result
            rephrased_query = classify_result.get('rephrased_query') if isinstance(classify_result, dict) else None
            # Step 3: Chuẩn bị query cho search
            if categories == ['small_talk']:
                print("💬 Small talk: Bỏ qua vector search, chỉ dùng LLM trả lời.")
                continue
            if not categories:
                print("⚠️  Không xác định được category, sẽ search toàn bộ database.")
            vector_query = rephrased_query if rephrased_query else query
            print(f"\n🔎 Query dùng cho vector search: {vector_query}")
            # Step 4: Semantic search (ChromaDB)
            print(f"\n🔍 BƯỚC 1: Semantic search (ChromaDB)")
            initial_results = 20 if categories else 10
            if categories:
                where_clause = {"category": {"$in": categories}}
                print(f"   📊 Sử dụng where clause: {where_clause}")
                results = collection.query(query_texts=[vector_query], n_results=initial_results, where=where_clause, include=["metadatas", "documents"])
            else:
                results = collection.query(query_texts=[vector_query], n_results=initial_results, include=["metadatas", "documents"])
            print(f"   ✅ Tìm thấy {len(results['documents'][0]) if results['documents'] else 0} kết quả từ ChromaDB")
            # Step 5: Hiển thị chi tiết kết quả semantic search
            if results['documents'] and results['documents'][0]:
                print(f"\n📋 Kết quả semantic search:")
                for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    category = meta.get('category', 'N/A')
                    doc_id = meta.get('doc_id', 'N/A')
                    source = meta.get('source', 'N/A')
                    preview = doc[:150].replace('\n', ' ') + "..." if len(doc) > 150 else doc
                    match_indicator = "✅" if not categories or category in categories else "❌"
                    print(f"  {i+1:2d}. {match_indicator} [{category}] {doc_id} - {source}")
                    print(f"      📄 {preview}\n")
            # Step 6: Chuẩn bị candidates
            candidate_content = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    document_text = results['documents'][0][i]
                    candidate_content.append({
                        'document': document_text,
                        'metadata': metadata,
                        'relevance_score': 0.0,
                        'original_rank': i + 1
                    })
            print(f"   ✅ Candidates cho BM25 reranking: {len(candidate_content)}")
            # Step 7: BM25 rerank
            if candidate_content:
                print(f"\n🔄 BƯỚC 2: BM25 Reranking")
                documents = [item['document'] for item in candidate_content]
                bm25 = BM25(documents)
                bm25_scores = bm25.get_scores(vector_query)
                for i, score in enumerate(bm25_scores):
                    candidate_content[i]['relevance_score'] = score
                candidate_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                print(f"\n🏆 Thứ tự sau khi BM25 rerank:")
                # Tính độ khớp với query để đánh dấu 🔥
                def is_relevant(doc, query):
                    query_words = set(query.lower().split())
                    doc_words = set(doc.lower().split())
                    # Đánh dấu 🔥 nếu có ít nhất 2 từ chung hoặc query là substring của doc
                    return len(query_words & doc_words) >= 2 or query.lower() in doc.lower()
                for i, item in enumerate(candidate_content):
                    meta = item['metadata']
                    original_rank = item['original_rank']
                    score = item['relevance_score']
                    doc_id = meta.get('doc_id', 'N/A')
                    source = meta.get('source', 'N/A')
                    rank_change = ""
                    if i + 1 < original_rank:
                        rank_change = f"📈 (tăng {original_rank - (i + 1)} bậc)"
                    elif i + 1 > original_rank:
                        rank_change = f"📉 (giảm {(i + 1) - original_rank} bậc)"
                    else:
                        rank_change = "➡️ (giữ nguyên)"
                    fire_flag = "🔥" if is_relevant(item['document'], vector_query) else "  "
                    print(f"   {i+1}. Score: {score:.4f} {fire_flag} | [{meta.get('category', 'N/A')}] {doc_id} - {source} {rank_change}")
                    # preview = item['document'][:100].replace('\n', ' ') + "..."
                    # print(f"      {preview}\n")
                print(f"\n🎯 BƯỚC 3: Kết quả cuối cùng (top 4)")
                print("-" * 50)
                for i, item in enumerate(candidate_content[:4]):
                    meta = item['metadata']
                    score = item['relevance_score']
                    doc_id = meta.get('doc_id', 'N/A')
                    source = meta.get('source', 'N/A')
                    preview = item['document'][:250].replace('\n', ' ') + "..." if len(item['document']) > 250 else item['document']
                    print(f"🥇 Kết quả {i+1}:")
                    print(f"   📊 BM25 Score: {score:.4f}")
                    print(f"   📁 Doc ID: {doc_id}")
                    print(f"   📄 Source: {source}")
                    print(f"   📝 Nội dung: {preview}\n")
                print(f"⏱️  Thời gian xử lý: {time.time() - start:.4f}s")
                print(f"📈 Tổng kết: {len(results['documents'][0]) if results['documents'] else 0} candidates → {len(candidate_content[:4])} final results")
            else:
                print("❌ Không có kết quả nào")
        
        elif choice == '3':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
        print("\n" + "="*80)