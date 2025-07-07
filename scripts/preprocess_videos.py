# /scripts/preprocess_videos.py
"""
Kịch bản chính để chạy pipeline tiền xử lý video (Phase 1).

Kịch bản này sẽ:
- Nhận một video đầu vào.
- Sử dụng VideoAnalysisOrchestrator để phân tích video theo từng cảnh logic.
- Trích xuất các vector đặc trưng đa phương thức cho mỗi cảnh.
- Lưu kết quả ra một file JSON.
"""
import asyncio
import argparse
import logging
import json
import os
from pathlib import Path
import sys

# Thêm đường dẫn gốc của dự án vào sys.path để có thể import các module
# từ thư mục `tools`.
# Giả sử kịch bản này được chạy từ thư mục gốc của dự án.
# Ví dụ: python scripts/preprocess_videos.py --input ...
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from tools.processors.orchestrator import VideoAnalysisOrchestrator

# Cấu hình logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


async def main(args):
    """
    Hàm chính để điều phối quá trình xử lý.
    """
    logger.info(f"Bắt đầu quá trình tiền xử lý cho file: {args.input}")

    # Kiểm tra xem file video có tồn tại không
    video_path = Path(args.input)
    if not video_path.is_file():
        logger.error(f"Lỗi: File video không tồn tại tại đường dẫn '{args.input}'")
        return

    # Tạo một ID duy nhất cho video từ tên file
    video_id = video_path.stem

    # Khởi tạo bộ điều phối
    orchestrator = VideoAnalysisOrchestrator()

    try:
        # Chạy pipeline xử lý
        analysis_results = await orchestrator.process_video_by_scenes(
            video_path=str(video_path),
            video_id=video_id
        )

        if not analysis_results:
            logger.warning(f"Không có kết quả nào được tạo ra cho video '{video_id}'.")
            return

        # Lưu kết quả ra file JSON
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{video_id}_analysis_results.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=4)

        logger.info(f"Thành công! Kết quả phân tích đã được lưu tại: {output_file}")
        logger.info(f"Tổng số cảnh được xử lý: {len(analysis_results)}")

        # In ra thông tin của cảnh đầu tiên để kiểm tra nhanh
        if analysis_results:
            first_scene = analysis_results[0]
            print("\n--- Thông tin của cảnh đầu tiên ---")
            print(f"  Scene ID: {first_scene['scene_id']}")
            print(f"  Thời gian: {first_scene['start_seconds']:.2f}s - {first_scene['end_seconds']:.2f}s")
            print(f"  Kích thước Visual Embedding: {len(first_scene['visual_embedding'] or [])}")
            print(f"  Kích thước Audio Embedding: {len(first_scene['audio_embedding'] or [])}")
            print(f"  Kích thước OCR Embedding: {len(first_scene['ocr_embedding'] or [])}")
            print(f"  Văn bản OCR: '{first_scene['raw_ocr_text']}'")


    except Exception as e:
        logger.error(f"Một lỗi không mong muốn đã xảy ra trong quá trình xử lý: {e}", exc_info=True)


if __name__ == "__main__":
    # Thiết lập trình phân tích cú pháp đối số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Chạy pipeline tiền xử lý video để trích xuất các vector đặc trưng đa phương thức."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Đường dẫn đến file video cần xử lý."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/processed_results",
        help="Thư mục để lưu file JSON chứa kết quả phân tích."
    )

    # Tạo một file video giả để chạy thử nếu chưa có
    # (Yêu cầu ffmpeg đã được cài đặt và có trong PATH)
    if not os.path.exists("dummy_video.mp4"):
        logger.info("Chưa có file video mẫu. Đang tạo 'dummy_video.mp4'...")
        try:
            os.system("ffmpeg -f lavfi -i testsrc=duration=15:size=320x240:rate=15 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=16000 -c:v libx264 -c:a aac -shortest dummy_video.mp4 -y")
        except Exception as e:
            logger.warning(f"Không thể tạo video giả, bạn cần cung cấp file video qua tham số --input. Lỗi: {e}")


    # Ví dụ cách chạy từ dòng lệnh:
    # python scripts/preprocess_videos.py --input dummy_video.mp4 --output ./my_results
    
    args = parser.parse_args()
    
    # Chạy hàm main bất đồng bộ
    asyncio.run(main(args))
