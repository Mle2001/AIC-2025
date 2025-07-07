# /scripts/migrate_database.py
"""
Kịch bản để nạp dữ liệu phân tích video (từ các file JSON)
vào cơ sở dữ liệu vector LanceDB.

Kịch bản này thực hiện các nhiệm vụ sau:
1.  Kết nối tới LanceDB.
2.  Định nghĩa và tạo bảng (table) 'scenes' nếu chưa tồn tại,
    dựa trên lược đồ (schema) đã được đặc tả.
3.  Quét một thư mục chứa các file JSON kết quả từ 'preprocess_videos.py'.
4.  Đọc từng file JSON và nạp (insert) dữ liệu vào bảng 'scenes'.
"""
import asyncio
import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import lancedb
import pyarrow as pa

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def get_db_schema() -> pa.Schema:
    """
    Định nghĩa và trả về lược đồ (schema) cho bảng 'scenes' trong LanceDB.
    Lược đồ này tuân thủ theo "Đặc tả Kỹ thuật: Cấu trúc Dữ liệu Vector Embedding".

    Returns:
        pa.Schema: Đối tượng schema của PyArrow.
    """
    # Định nghĩa kiểu dữ liệu cho các cột vector với số chiều tương ứng
    visual_embedding_type = pa.list_(pa.float32(), 4096)
    audio_embedding_type = pa.list_(pa.float32(), 512)
    ocr_embedding_type = pa.list_(pa.float32(), 384)

    schema = pa.schema([
        pa.field("scene_id", pa.string()),
        pa.field("video_id", pa.string()),
        pa.field("start_seconds", pa.float32()),
        pa.field("end_seconds", pa.float32()),
        pa.field("visual_embedding", visual_embedding_type),
        pa.field("audio_embedding", audio_embedding_type),
        pa.field("ocr_embedding", ocr_embedding_type),
        pa.field("raw_ocr_text", pa.string())
    ])
    return schema

async def migrate_data_to_db(db_path: str, input_dir: str):
    """
    Hàm chính thực hiện việc kết nối DB, đọc file và nạp dữ liệu.

    Args:
        db_path (str): Đường dẫn đến thư mục lưu trữ của LanceDB.
        input_dir (str): Đường dẫn đến thư mục chứa các file JSON kết quả.
    """
    logger.info(f"Bắt đầu quá trình nạp dữ liệu vào LanceDB tại: {db_path}")
    
    try:
        # Kết nối hoặc tạo mới cơ sở dữ liệu
        db = lancedb.connect(db_path)
        
        # Lấy lược đồ và tên bảng
        table_name = "scenes"
        schema = get_db_schema()

        # Kiểm tra xem bảng đã tồn tại chưa, nếu chưa thì tạo mới
        if table_name not in db.table_names():
            logger.info(f"Bảng '{table_name}' chưa tồn tại. Đang tạo bảng mới...")
            db.create_table(table_name, schema=schema)
            logger.info(f"Đã tạo bảng '{table_name}' thành công.")
        
        table = db.open_table(table_name)
        
        # Quét thư mục đầu vào để tìm các file JSON
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))

        if not json_files:
            logger.warning(f"Không tìm thấy file JSON nào trong thư mục: {input_dir}")
            return

        logger.info(f"Tìm thấy {len(json_files)} file JSON để xử lý.")

        total_scenes_added = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data_to_add = json.load(f)
                
                if not data_to_add:
                    logger.warning(f"File {json_file.name} rỗng, bỏ qua.")
                    continue

                # LanceDB có thể thêm trực tiếp một list các dictionary
                table.add(data_to_add)
                total_scenes_added += len(data_to_add)
                logger.info(f"Đã nạp thành công {len(data_to_add)} cảnh từ file {json_file.name}.")

            except json.JSONDecodeError:
                logger.error(f"Lỗi khi đọc file JSON: {json_file.name}. File có thể bị hỏng.")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý file {json_file.name}: {e}", exc_info=True)

        logger.info("--- Quá trình nạp dữ liệu hoàn tất ---")
        logger.info(f"Tổng số cảnh đã được thêm vào DB: {total_scenes_added}")
        logger.info(f"Tổng số bản ghi trong bảng '{table_name}': {len(table)}")

    except Exception as e:
        logger.error(f"Một lỗi nghiêm trọng đã xảy ra trong quá trình migrate: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nạp dữ liệu video đã phân tích từ JSON vào LanceDB."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="./data/processed_results",
        help="Thư mục chứa các file JSON kết quả từ pipeline tiền xử lý."
    )
    parser.add_argument(
        "-d", "--db-path",
        type=str,
        default="./data/lancedb",
        help="Đường dẫn đến thư mục lưu trữ của LanceDB."
    )

    # Ví dụ cách chạy từ dòng lệnh:
    # python scripts/migrate_database.py --input-dir ./data/processed_results --db-path ./data/lancedb
    
    args = parser.parse_args()
    
    # Chạy hàm main bất đồng bộ
    asyncio.run(migrate_data_to_db(args.db_path, args.input_dir))
