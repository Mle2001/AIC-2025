# /scripts/benchmark.py
"""
Kịch bản để đo lường và đánh giá hiệu năng (benchmark) của hệ thống truy xuất.

Kịch bản này sẽ:
1.  Định nghĩa một tập hợp các truy vấn mẫu.
2.  Mô phỏng luồng xử lý của Agent: Phân tích truy vấn và sau đó truy xuất.
3.  Đo lường các chỉ số hiệu năng chính:
    - Latency (Độ trễ): Thời gian phản hồi cho một truy vấn.
    - Throughput (Thông lượng): Số truy vấn mỗi giây (QPS).
4.  In ra báo cáo tổng hợp.
"""
import asyncio
import argparse
import logging
import time
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Any

# Thêm đường dẫn gốc của dự án
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import các thành phần cần thiết từ hệ thống
from tools.retrieval import HybridRetriever
# Mô phỏng output của QueryUnderstandingAgent
from agents.conversational.query_understanding_agent import StructuredQuery

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Định nghĩa các truy vấn mẫu để benchmark ---
BENCHMARK_QUERIES = [
    {
        "description": "Tìm kiếm ngữ nghĩa đơn giản",
        "query_text": "một người đang chơi đàn guitar trên bãi biển",
        "entities": ["đàn guitar", "bãi biển"]
    },
    {
        "description": "Tìm kiếm thực thể cụ thể",
        "query_text": "video có sự xuất hiện của tổng thống",
        "entities": ["tổng thống"]
    },
    {
        "description": "Truy vấn có chứa từ khóa OCR",
        "query_text": "cảnh có chữ 'công thức nấu ăn'",
        "entities": ["công thức nấu ăn"]
    },
    {
        "description": "Truy vấn phức hợp",
        "query_text": "tìm cảnh phim hành động có cháy nổ và ô tô rượt đuổi",
        "entities": ["phim hành động", "cháy nổ", "ô tô"]
    }
]

class BenchmarkRunner:
    """
    Lớp chịu trách nhiệm chạy các bài kiểm tra hiệu năng.
    """
    def __init__(self):
        """Khởi tạo các công cụ cần thiết."""
        logger.info("Đang khởi tạo các công cụ cho benchmark...")
        try:
            self.hybrid_retriever = HybridRetriever()
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo retriever. Hãy đảm bảo DB đã được migrate. Lỗi: {e}")
            sys.exit(1)
        self.latencies: List[float] = []
        self.qps_results: List[int] = []

    async def _run_single_query(self, query_info: Dict[str, Any]) -> float:
        """Chạy một truy vấn duy nhất và trả về độ trễ."""
        query_text = query_info["query_text"]
        entities = query_info["entities"]
        
        # Mô phỏng output của QueryUnderstandingAgent
        # Trong thực tế, bước này cũng sẽ tốn thời gian, nhưng ở đây ta tập trung
        # vào hiệu năng của tầng truy xuất.
        
        start_time = time.perf_counter()
        
        # Gọi công cụ truy xuất
        results = await self.hybrid_retriever.retrieve_comprehensive(
            query_text=query_text,
            entities=entities,
            top_k=10
        )
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        logger.info(f"Query '{query_text[:30]}...' | Results: {len(results)} | Latency: {latency:.4f}s")
        return latency

    async def run_latency_test(self):
        """
        Chạy bài kiểm tra độ trễ (latency).
        Thực hiện tuần tự từng truy vấn để đo thời gian phản hồi chính xác.
        """
        logger.info("\n" + "="*20 + " BẮT ĐẦU KIỂM TRA ĐỘ TRỄ (LATENCY) " + "="*20)
        self.latencies = []
        for query_info in BENCHMARK_QUERIES:
            latency = await self._run_single_query(query_info)
            self.latencies.append(latency)
            await asyncio.sleep(0.5) # Nghỉ ngắn giữa các truy vấn

    async def run_throughput_test(self, duration: int, concurrent_tasks: int):
        """
        Chạy bài kiểm tra thông lượng (throughput).
        Thực hiện nhiều truy vấn đồng thời để xem hệ thống xử lý được bao nhiêu QPS.
        """
        logger.info("\n" + "="*20 + f" BẮT ĐẦU KIỂM TRA THÔNG LƯỢNG (THROUGHPUT) trong {duration}s " + "="*20)
        
        query_info = BENCHMARK_QUERIES[0] # Dùng một truy vấn đơn giản để test
        total_queries_done = 0
        start_time = time.time()

        async def worker():
            nonlocal total_queries_done
            while time.time() - start_time < duration:
                await self._run_single_query(query_info)
                total_queries_done += 1

        tasks = [worker() for _ in range(concurrent_tasks)]
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        qps = total_queries_done / actual_duration if actual_duration > 0 else 0
        self.qps_results.append(qps)

    def print_summary(self):
        """In báo cáo tổng hợp kết quả benchmark."""
        print("\n" + "="*25 + " BÁO CÁO KẾT QUẢ BENCHMARK " + "="*25)
        
        if self.latencies:
            print("\n--- Kết quả Độ trễ (Latency) ---")
            print(f"  - Số lượng truy vấn test: {len(self.latencies)}")
            print(f"  - Độ trễ trung bình:     {statistics.mean(self.latencies):.4f} giây")
            print(f"  - Độ trễ trung vị:       {statistics.median(self.latencies):.4f} giây")
            print(f"  - Độ trễ thấp nhất (nhanh nhất): {min(self.latencies):.4f} giây")
            print(f"  - Độ trễ cao nhất (chậm nhất):  {max(self.latencies):.4f} giây")
            print(f"  - Độ lệch chuẩn:         {statistics.stdev(self.latencies):.4f} giây")

        if self.qps_results:
            print("\n--- Kết quả Thông lượng (Throughput) ---")
            avg_qps = statistics.mean(self.qps_results)
            print(f"  - Thông lượng trung bình: {avg_qps:.2f} Queries Per Second (QPS)")

        print("\n" + "="*75)


async def main(args):
    runner = BenchmarkRunner()

    if args.test_type in ['all', 'latency']:
        await runner.run_latency_test()
    
    if args.test_type in ['all', 'throughput']:
        await runner.run_throughput_test(duration=args.duration, concurrent_tasks=args.concurrency)

    runner.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy benchmark cho hệ thống truy xuất video.")
    parser.add_argument(
        "--test-type",
        type=str,
        default="all",
        choices=["latency", "throughput", "all"],
        help="Loại bài test cần chạy."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Thời gian (giây) để chạy bài test thông lượng."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Số lượng tác vụ đồng thời để chạy bài test thông lượng."
    )

    # Ví dụ cách chạy:
    # python scripts/benchmark.py --test-type all --duration 15 --concurrency 20
    
    args = parser.parse_args()
    asyncio.run(main(args))

