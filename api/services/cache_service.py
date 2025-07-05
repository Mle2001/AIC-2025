# api/services/cache_service.py
"""
Cache Service - Business logic cho caching system
Dev2: API Integration & Services - quản lý cache cho performance optimization
Current: 2025-07-03 14:12:27 UTC, User: xthanh1910
"""

import asyncio
import time
import json
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import redis.asyncio as redis
import uuid

# Import để serialize/deserialize các objects phức tạp
from agents.conversational.context_manager_agent import SessionContext
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    print("Warning: msgpack not available, using json fallback for serialization")

class CacheService:
    """
    Service quản lý cache cho toàn bộ hệ thống
    Dev2: Tối ưu performance bằng caching cho tất cả services
    """

    def __init__(self):
        """
        Khởi tạo CacheService
        """
        # Redis connection (production)
        self.redis_client = None
        self.redis_config = {
            'host': 'localhost',  # Dev3 sẽ cung cấp config thật
            'port': 6379,
            'db': 0,
            'password': None,
            'decode_responses': False,  # Để handle binary data
            'socket_connect_timeout': 5,
            'socket_timeout': 5
        }

        # In-memory cache (fallback khi Redis không có)
        self._memory_cache = {}
        self._memory_cache_ttl = {}  # key -> expiration_time
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }

        # Cache configuration
        self.default_ttl = 3600  # 1 hour default TTL
        self.max_memory_cache_size = 10000  # Max items in memory cache
        self.key_prefix = "ai_challenge:"  # Prefix cho tất cả keys
        self.use_compression = True  # Compress large values
        self.compression_threshold = 1024  # Compress if > 1KB

        # Cache namespaces cho các loại data khác nhau
        self.namespaces = {
            'session': 'session:',
            'user': 'user:',
            'video': 'video:',
            'job': 'job:',
            'temp': 'temp:',
            'api': 'api:',
            'system': 'system:'
        }

        print(f"[{datetime.utcnow()}] CacheService initialized by user: xthanh1910")

    # ================================
    # CONNECTION MANAGEMENT
    # ================================

    async def initialize(self):
        """
        Khởi tạo cache connections
        """
        try:
            # Thử kết nối Redis trước
            await self._connect_redis()

            # Start background cleanup task cho memory cache
            asyncio.create_task(self._memory_cache_cleanup_task())

            print(f"[{datetime.utcnow()}] CacheService initialized successfully")

        except Exception as e:
            print(f"[{datetime.utcnow()}] CacheService initialization warning: {str(e)}")
            print("[{datetime.utcnow()}] Falling back to memory cache only")

    async def _connect_redis(self):
        """
        Kết nối tới Redis server
        """
        try:
            self.redis_client = redis.Redis(**self.redis_config)

            # Test connection
            await self.redis_client.ping()
            print(f"[{datetime.utcnow()}] Connected to Redis successfully")

        except Exception as e:
            print(f"Redis connection failed: {str(e)}")
            self.redis_client = None
            raise

    async def close(self):
        """
        Đóng connections
        """
        try:
            if self.redis_client:
                await self.redis_client.close()
                print(f"[{datetime.utcnow()}] Redis connection closed")

            # Clear memory cache
            self._memory_cache.clear()
            self._memory_cache_ttl.clear()

            print(f"[{datetime.utcnow()}] CacheService shutdown completed")

        except Exception as e:
            print(f"Error closing cache service: {str(e)}")

    # ================================
    # CORE CACHE OPERATIONS
    # ================================

    async def get(self, key: str, namespace: str = None) -> Any:
        """
        Lấy value từ cache theo key
        """
        try:
            full_key = self._build_key(key, namespace)

            # Thử Redis trước
            if self.redis_client:
                try:
                    value = await self.redis_client.get(full_key)
                    if value is not None:
                        self._cache_stats['hits'] += 1
                        return self._deserialize_value(value)
                except Exception as e:
                    print(f"Redis get error: {str(e)}")
                    self._cache_stats['errors'] += 1

            # Fallback to memory cache
            if full_key in self._memory_cache:
                # Kiểm tra TTL
                if self._is_memory_key_expired(full_key):
                    del self._memory_cache[full_key]
                    del self._memory_cache_ttl[full_key]
                    self._cache_stats['misses'] += 1
                    return None

                self._cache_stats['hits'] += 1
                return self._memory_cache[full_key]

            self._cache_stats['misses'] += 1
            return None

        except Exception as e:
            print(f"Error getting cache key {key}: {str(e)}")
            self._cache_stats['errors'] += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        namespace: str = None
    ) -> bool:
        """
        Lưu value vào cache với TTL
        """
        try:
            full_key = self._build_key(key, namespace)
            ttl = ttl or self.default_ttl

            # Thử Redis trước
            if self.redis_client:
                try:
                    serialized_value = self._serialize_value(value)
                    await self.redis_client.setex(full_key, ttl, serialized_value)
                    self._cache_stats['sets'] += 1
                    return True
                except Exception as e:
                    print(f"Redis set error: {str(e)}")
                    self._cache_stats['errors'] += 1

            # Fallback to memory cache
            # Cleanup memory cache nếu quá lớn
            if len(self._memory_cache) >= self.max_memory_cache_size:
                await self._cleanup_memory_cache()

            # Set value và TTL
            self._memory_cache[full_key] = value
            self._memory_cache_ttl[full_key] = time.time() + ttl
            self._cache_stats['sets'] += 1

            return True

        except Exception as e:
            print(f"Error setting cache key {key}: {str(e)}")
            self._cache_stats['errors'] += 1
            return False

    async def delete(self, key: str, namespace: str = None) -> bool:
        """
        Xóa key khỏi cache
        """
        try:
            full_key = self._build_key(key, namespace)
            deleted = False

            # Delete từ Redis
            if self.redis_client:
                try:
                    result = await self.redis_client.delete(full_key)
                    deleted = result > 0
                except Exception as e:
                    print(f"Redis delete error: {str(e)}")
                    self._cache_stats['errors'] += 1

            # Delete từ memory cache
            if full_key in self._memory_cache:
                del self._memory_cache[full_key]
                deleted = True

            if full_key in self._memory_cache_ttl:
                del self._memory_cache_ttl[full_key]

            if deleted:
                self._cache_stats['deletes'] += 1

            return deleted

        except Exception as e:
            print(f"Error deleting cache key {key}: {str(e)}")
            self._cache_stats['errors'] += 1
            return False

    async def exists(self, key: str, namespace: str = None) -> bool:
        """
        Kiểm tra key có tồn tại trong cache không
        """
        try:
            full_key = self._build_key(key, namespace)

            # Check Redis
            if self.redis_client:
                try:
                    exists = await self.redis_client.exists(full_key)
                    if exists:
                        return True
                except Exception as e:
                    print(f"Redis exists error: {str(e)}")

            # Check memory cache
            if full_key in self._memory_cache:
                if not self._is_memory_key_expired(full_key):
                    return True
                else:
                    # Clean up expired key
                    del self._memory_cache[full_key]
                    del self._memory_cache_ttl[full_key]

            return False

        except Exception as e:
            print(f"Error checking cache key existence {key}: {str(e)}")
            return False

    async def expire(self, key: str, ttl: int, namespace: str = None) -> bool:
        """
        Set TTL cho existing key
        """
        try:
            full_key = self._build_key(key, namespace)

            # Set TTL trong Redis
            if self.redis_client:
                try:
                    result = await self.redis_client.expire(full_key, ttl)
                    if result:
                        return True
                except Exception as e:
                    print(f"Redis expire error: {str(e)}")

            # Set TTL trong memory cache
            if full_key in self._memory_cache:
                self._memory_cache_ttl[full_key] = time.time() + ttl
                return True

            return False

        except Exception as e:
            print(f"Error setting TTL for cache key {key}: {str(e)}")
            return False

    # ================================
    # SPECIALIZED CACHE OPERATIONS
    # ================================

    async def get_or_set(
        self,
        key: str,
        value_func: callable,
        ttl: int = None,
        namespace: str = None
    ) -> Any:
        """
        Lấy value từ cache, nếu không có thì compute và cache
        """
        try:
            # Thử lấy từ cache trước
            cached_value = await self.get(key, namespace)
            if cached_value is not None:
                return cached_value

            # Không có trong cache, compute value
            if asyncio.iscoroutinefunction(value_func):
                computed_value = await value_func()
            else:
                computed_value = value_func()

            # Cache computed value
            if computed_value is not None:
                await self.set(key, computed_value, ttl, namespace)

            return computed_value

        except Exception as e:
            print(f"Error in get_or_set for key {key}: {str(e)}")
            # Fallback: return computed value without caching
            try:
                if asyncio.iscoroutinefunction(value_func):
                    return await value_func()
                else:
                    return value_func()
            except Exception:
                return None

    async def increment(self, key: str, amount: int = 1, namespace: str = None) -> int:
        """
        Increment integer value trong cache
        """
        try:
            full_key = self._build_key(key, namespace)

            # Try Redis first
            if self.redis_client:
                try:
                    result = await self.redis_client.incrby(full_key, amount)
                    return result
                except Exception as e:
                    print(f"Redis increment error: {str(e)}")

            # Fallback to memory cache
            current_value = await self.get(key, namespace) or 0
            new_value = int(current_value) + amount
            await self.set(key, new_value, namespace=namespace)

            return new_value

        except Exception as e:
            print(f"Error incrementing cache key {key}: {str(e)}")
            return 0

    async def decrement(self, key: str, amount: int = 1, namespace: str = None) -> int:
        """
        Decrement integer value trong cache
        """
        return await self.increment(key, -amount, namespace)

    async def set_many(
        self,
        data: Dict[str, Any],
        ttl: int = None,
        namespace: str = None
    ) -> bool:
        """
        Set nhiều key-value pairs cùng lúc
        """
        try:
            success_count = 0

            for key, value in data.items():
                if await self.set(key, value, ttl, namespace):
                    success_count += 1

            return success_count == len(data)

        except Exception as e:
            print(f"Error setting multiple cache keys: {str(e)}")
            return False

    async def get_many(
        self,
        keys: List[str],
        namespace: str = None
    ) -> Dict[str, Any]:
        """
        Lấy nhiều values cùng lúc
        """
        try:
            result = {}

            for key in keys:
                value = await self.get(key, namespace)
                if value is not None:
                    result[key] = value

            return result

        except Exception as e:
            print(f"Error getting multiple cache keys: {str(e)}")
            return {}

    async def delete_pattern(self, pattern: str, namespace: str = None) -> int:
        """
        Xóa tất cả keys match pattern
        """
        try:
            full_pattern = self._build_key(pattern, namespace)
            deleted_count = 0

            # Delete từ Redis
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys(full_pattern)
                    if keys:
                        deleted_count += await self.redis_client.delete(*keys)
                except Exception as e:
                    print(f"Redis pattern delete error: {str(e)}")

            # Delete từ memory cache
            keys_to_delete = [
                key for key in self._memory_cache.keys()
                if self._match_pattern(key, full_pattern)
            ]

            for key in keys_to_delete:
                del self._memory_cache[key]
                if key in self._memory_cache_ttl:
                    del self._memory_cache_ttl[key]
                deleted_count += 1

            return deleted_count

        except Exception as e:
            print(f"Error deleting pattern {pattern}: {str(e)}")
            return 0

    # ================================
    # SESSION-SPECIFIC METHODS
    # ================================

    async def save_session_context(
        self,
        session_id: str,
        context: SessionContext,
        ttl: int = None
    ) -> bool:
        """
        Lưu session context vào cache
        """
        try:
            # Convert SessionContext object to dict để cache
            context_dict = context.dict() if hasattr(context, 'dict') else context.__dict__

            return await self.set(
                key=session_id,
                value=context_dict,
                ttl=ttl or 3600,  # 1 hour default cho sessions
                namespace='session'
            )

        except Exception as e:
            print(f"Error saving session context: {str(e)}")
            return False

    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """
        Lấy session context từ cache
        """
        try:
            context_dict = await self.get(session_id, namespace='session')

            if context_dict:
                # Convert dict back to SessionContext object
                return SessionContext(**context_dict)

            return None

        except Exception as e:
            print(f"Error getting session context: {str(e)}")
            return None

    async def delete_session_context(self, session_id: str) -> bool:
        """
        Xóa session context
        """
        return await self.delete(session_id, namespace='session')

    async def extend_session_ttl(self, session_id: str, ttl: int = 3600) -> bool:
        """
        Gia hạn TTL cho session
        """
        return await self.expire(session_id, ttl, namespace='session')

    # ================================
    # API RESPONSE CACHING
    # ================================

    async def cache_api_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response_data: Any,
        ttl: int = 300  # 5 minutes default
    ) -> bool:
        """
        Cache API response để tránh duplicate calls
        """
        try:
            # Tạo cache key từ endpoint + params
            cache_key = self._generate_api_cache_key(endpoint, params)

            return await self.set(
                key=cache_key,
                value=response_data,
                ttl=ttl,
                namespace='api'
            )

        except Exception as e:
            print(f"Error caching API response: {str(e)}")
            return False

    async def get_cached_api_response(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        Lấy cached API response
        """
        try:
            cache_key = self._generate_api_cache_key(endpoint, params)
            return await self.get(cache_key, namespace='api')

        except Exception as e:
            print(f"Error getting cached API response: {str(e)}")
            return None

    def _generate_api_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Tạo cache key cho API response
        """
        # Sort params để consistent key
        sorted_params = json.dumps(params, sort_keys=True)
        hash_input = f"{endpoint}:{sorted_params}"

        # Hash để tránh key quá dài
        return hashlib.md5(hash_input.encode()).hexdigest()

    # ================================
    # TEMP DATA CACHING
    # ================================

    async def cache_temp_data(
        self,
        data: Any,
        ttl: int = 1800,  # 30 minutes default
        prefix: str = "temp"
    ) -> str:
        """
        Cache temporary data và return cache key
        """
        try:
            # Generate unique temp key
            temp_key = f"{prefix}_{uuid.uuid4().hex[:8]}"

            success = await self.set(
                key=temp_key,
                value=data,
                ttl=ttl,
                namespace='temp'
            )

            return temp_key if success else None

        except Exception as e:
            print(f"Error caching temp data: {str(e)}")
            return None

    async def get_temp_data(self, temp_key: str) -> Any:
        """
        Lấy temporary data theo key
        """
        return await self.get(temp_key, namespace='temp')

    async def delete_temp_data(self, temp_key: str) -> bool:
        """
        Xóa temporary data
        """
        return await self.delete(temp_key, namespace='temp')

    # ================================
    # STATISTICS & MONITORING
    # ================================

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Lấy cache statistics
        """
        try:
            stats = {
                'memory_cache': {
                    'size': len(self._memory_cache),
                    'max_size': self.max_memory_cache_size,
                    'usage_percent': (len(self._memory_cache) / self.max_memory_cache_size) * 100
                },
                'operations': dict(self._cache_stats),
                'hit_rate': 0,
                'redis_connected': self.redis_client is not None
            }

            # Calculate hit rate
            total_gets = stats['operations']['hits'] + stats['operations']['misses']
            if total_gets > 0:
                stats['hit_rate'] = (stats['operations']['hits'] / total_gets) * 100

            # Redis-specific stats nếu có connection
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info()
                    stats['redis'] = {
                        'used_memory': redis_info.get('used_memory_human', 'unknown'),
                        'connected_clients': redis_info.get('connected_clients', 0),
                        'total_commands_processed': redis_info.get('total_commands_processed', 0),
                        'keyspace_hits': redis_info.get('keyspace_hits', 0),
                        'keyspace_misses': redis_info.get('keyspace_misses', 0)
                    }

                    # Redis hit rate
                    redis_hits = redis_info.get('keyspace_hits', 0)
                    redis_misses = redis_info.get('keyspace_misses', 0)
                    redis_total = redis_hits + redis_misses

                    if redis_total > 0:
                        stats['redis']['hit_rate'] = (redis_hits / redis_total) * 100

                except Exception as e:
                    stats['redis'] = {'error': str(e)}

            return stats

        except Exception as e:
            return {'error': str(e)}

    async def clear_all_cache(self) -> Dict[str, Any]:
        """
        Xóa toàn bộ cache (emergency function)
        """
        try:
            result = {
                'redis_cleared': False,
                'memory_cleared': False,
                'errors': []
            }

            # Clear Redis
            if self.redis_client:
                try:
                    await self.redis_client.flushdb()
                    result['redis_cleared'] = True
                except Exception as e:
                    result['errors'].append(f"Redis clear error: {str(e)}")

            # Clear memory cache
            try:
                self._memory_cache.clear()
                self._memory_cache_ttl.clear()
                result['memory_cleared'] = True
            except Exception as e:
                result['errors'].append(f"Memory clear error: {str(e)}")

            # Reset stats
            self._cache_stats = {
                'hits': 0,
                'misses': 0,
                'sets': 0,
                'deletes': 0,
                'errors': 0
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    async def emergency_clear_cache(self) -> Dict[str, Any]:
        """
        Emergency cache clear cho admin
        """
        print(f"[{datetime.utcnow()}] Emergency cache clear initiated by xthanh1910")
        return await self.clear_all_cache()

    # ================================
    # HELPER METHODS
    # ================================

    def _build_key(self, key: str, namespace: str = None) -> str:
        """
        Build full cache key với prefix và namespace
        """
        if namespace and namespace in self.namespaces:
            namespace_prefix = self.namespaces[namespace]
        else:
            namespace_prefix = namespace + ":" if namespace else ""

        return f"{self.key_prefix}{namespace_prefix}{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """
        Serialize value để lưu vào cache
        """
        try:
            # Use msgpack nếu có, nhanh hơn pickle
            if MSGPACK_AVAILABLE:
                serialized = msgpack.packb(value, use_bin_type=True)
            else:
                serialized = pickle.dumps(value)

            # Compress nếu value lớn
            if self.use_compression and len(serialized) > self.compression_threshold:
                import gzip
                serialized = gzip.compress(serialized)
                # Thêm marker để biết data đã compressed
                serialized = b'COMPRESSED:' + serialized

            return serialized

        except Exception as e:
            print(f"Serialization error: {str(e)}")
            # Fallback to json for simple types
            try:
                return json.dumps(value).encode('utf-8')
            except Exception:
                return str(value).encode('utf-8')

    def _deserialize_value(self, serialized: bytes) -> Any:
        """
        Deserialize value từ cache
        """
        try:
            # Check if compressed
            if serialized.startswith(b'COMPRESSED:'):
                import gzip
                serialized = gzip.decompress(serialized[11:])  # Remove 'COMPRESSED:' prefix

            # Try msgpack first
            if MSGPACK_AVAILABLE:
                try:
                    return msgpack.unpackb(serialized, raw=False)
                except Exception:
                    pass

            # Try pickle
            try:
                return pickle.loads(serialized)
            except Exception:
                pass

            # Fallback to json
            try:
                return json.loads(serialized.decode('utf-8'))
            except Exception:
                pass

            # Last resort: return as string
            return serialized.decode('utf-8')

        except Exception as e:
            print(f"Deserialization error: {str(e)}")
            return None

    def _is_memory_key_expired(self, full_key: str) -> bool:
        """
        Kiểm tra memory cache key đã hết hạn chưa
        """
        if full_key not in self._memory_cache_ttl:
            return False

        return time.time() > self._memory_cache_ttl[full_key]

    async def _cleanup_memory_cache(self):
        """
        Cleanup expired keys từ memory cache
        """
        try:
            current_time = time.time()
            expired_keys = []

            for key, expiry_time in self._memory_cache_ttl.items():
                if current_time > expiry_time:
                    expired_keys.append(key)

            # Remove expired keys
            for key in expired_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                del self._memory_cache_ttl[key]

            # If still too large, remove oldest entries
            if len(self._memory_cache) >= self.max_memory_cache_size:
                # Sort by expiry time và remove earliest expiring
                sorted_keys = sorted(
                    self._memory_cache_ttl.items(),
                    key=lambda x: x[1]
                )

                # Remove 20% of cache
                remove_count = self.max_memory_cache_size // 5
                for key, _ in sorted_keys[:remove_count]:
                    if key in self._memory_cache:
                        del self._memory_cache[key]
                    if key in self._memory_cache_ttl:
                        del self._memory_cache_ttl[key]

        except Exception as e:
            print(f"Error cleaning up memory cache: {str(e)}")

    async def _memory_cache_cleanup_task(self):
        """
        Background task để cleanup memory cache định kỳ
        """
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_memory_cache()
            except Exception as e:
                print(f"Memory cache cleanup task error: {str(e)}")
                await asyncio.sleep(60)  # Retry after 1 minute on error

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """
        Simple pattern matching cho delete_pattern
        """
        try:
            # Basic wildcard matching with *
            if '*' not in pattern:
                return key == pattern

            # Split pattern by *
            parts = pattern.split('*')
            key_pos = 0

            for i, part in enumerate(parts):
                if not part:  # Empty part từ **
                    continue

                if i == 0:  # First part phải match từ đầu
                    if not key.startswith(part):
                        return False
                    key_pos = len(part)
                elif i == len(parts) - 1:  # Last part phải match cuối
                    if not key.endswith(part):
                        return False
                else:  # Middle parts
                    pos = key.find(part, key_pos)
                    if pos == -1:
                        return False
                    key_pos = pos + len(part)

            return True

        except Exception:
            return False

    # ================================
    # HEALTH CHECK
    # ================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check cho CacheService
        """
        try:
            start_time = time.time()

            health_data = {
                'service': 'cache_service',
                'status': 'unknown',
                'redis_available': False,
                'memory_cache_available': True,
                'response_time_ms': 0,
                'cache_stats': await self.get_cache_stats()
            }

            # Test Redis connection
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_data['redis_available'] = True
                except Exception as e:
                    health_data['redis_error'] = str(e)

            # Test cache operations
            test_key = f"health_check_{int(time.time())}"
            test_value = {'test': True, 'timestamp': datetime.utcnow().isoformat()}

            # Test set
            set_success = await self.set(test_key, test_value, ttl=60, namespace='system')

            # Test get
            retrieved_value = await self.get(test_key, namespace='system')
            get_success = retrieved_value is not None

            # Test delete
            delete_success = await self.delete(test_key, namespace='system')

            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            health_data['response_time_ms'] = round(response_time, 2)

            # Determine overall status
            if set_success and get_success and delete_success:
                if health_data['redis_available']:
                    health_data['status'] = 'healthy'
                else:
                    health_data['status'] = 'degraded'  # Memory cache only
            else:
                health_data['status'] = 'unhealthy'

            health_data['operations_test'] = {
                'set': set_success,
                'get': get_success,
                'delete': delete_success
            }

            return health_data

        except Exception as e:
            return {
                'service': 'cache_service',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Export service instance
cache_service = CacheService()