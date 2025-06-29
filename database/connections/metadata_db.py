# /database/connections/metadata_db.py
"""
PostgreSQL connection cho metadata storage sử dụng SQLAlchemy.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataDB:
    """
    Lớp quản lý kết nối và các thao tác với PostgreSQL cho việc lưu trữ
    metadata của video, người dùng, và các thông tin cấu trúc khác.
    """
    def __init__(self, database_url: str):
        """
        Khởi tạo kết nối tới PostgreSQL.

        Args:
            database_url (str): URL kết nối theo định dạng của SQLAlchemy
                                (ví dụ: "postgresql+asyncpg://user:password@host:port/dbname").
        """
        try:
            self.engine = create_async_engine(database_url)
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info(f"Đã tạo engine kết nối tới PostgreSQL.")
        except Exception as e:
            logger.error(f"Lỗi khi tạo engine SQLAlchemy: {e}", exc_info=True)
            self.engine = None
            self.SessionLocal = None

    async def get_session(self) -> AsyncSession:
        """Cung cấp một session để tương tác với DB."""
        if not self.SessionLocal:
            raise ConnectionError("Kết nối PostgreSQL chưa được thiết lập.")
        return self.SessionLocal()

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Cập nhật sở thích của người dùng vào cơ sở dữ liệu.
        (Đây là một ví dụ, bạn sẽ cần tạo bảng 'users' trước).
        """
        async with self.get_session() as session:
            async with session.begin():
                # Giả sử bạn có một bảng 'users' với cột 'preferences' dạng JSONB
                # Câu lệnh SQL này chỉ là ví dụ
                stmt = text("""
                    UPDATE users
                    SET preferences = preferences || :new_prefs
                    WHERE user_id = :user_id
                """)
                await session.execute(stmt, {"new_prefs": json.dumps(preferences), "user_id": user_id})
        logger.info(f"Đã cập nhật preferences cho user: {user_id}")


    async def close(self):
        """Đóng engine kết nối."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Đã đóng engine kết nối PostgreSQL.")
