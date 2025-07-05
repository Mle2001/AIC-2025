# api/services/user_service.py
"""
User Service - Business logic cho user management system
Dev2: API Integration & Services - kết nối API với Dev1's agents và Dev3's database
Current: 2025-07-03 14:07:33 UTC, User: xthanh1910
"""

import asyncio
import time
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import bcrypt
import jwt

# Import database models/connections từ Dev3 (placeholder imports)
# Dev3 sẽ implement database thực tế cho user management
try:
    from database.models.user_models import User, UserProfile, UserActivity, UserSession
    from database.models.auth_models import UserAuth, LoginAttempt, PasswordReset
    from database.connections.user_db import UserDatabase
    from database.connections.auth_db import AuthDatabase
    from database.connections.activity_db import ActivityDatabase
except ImportError:
    # Fallback nếu Dev3 chưa implement
    print("Warning: User database models not found, using mock implementations")
    User = dict
    UserProfile = dict
    UserActivity = dict
    UserSession = dict
    UserAuth = dict
    LoginAttempt = dict
    PasswordReset = dict
    UserDatabase = None
    AuthDatabase = None
    ActivityDatabase = None

# Import cache service để lưu user sessions
from .cache_service import CacheService

class UserService:
    """
    Service xử lý logic user management và authentication
    Dev2 chỉ làm integration - không viết complex authentication logic
    """

    def __init__(self):
        """
        Khởi tạo UserService
        """
        self.cache_service = CacheService()

        # Database connections (Dev3's responsibility)
        self.user_db = UserDatabase() if UserDatabase else None
        self.auth_db = AuthDatabase() if AuthDatabase else None
        self.activity_db = ActivityDatabase() if ActivityDatabase else None

        # Mock storage nếu database chưa có
        self._mock_users = {}           # user_id -> user_data
        self._mock_profiles = {}        # user_id -> profile_data
        self._mock_activities = {}      # user_id -> [activities]
        self._mock_sessions = {}        # session_token -> session_data
        self._mock_auth = {}           # user_id -> auth_data

        # Service configuration
        self.jwt_secret = "dev_secret_key_change_in_production"  # Dev3 sẽ cung cấp thật
        self.jwt_algorithm = "HS256"
        self.jwt_expire_hours = 24
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.password_min_length = 8

        # Default admin user cho development
        self._create_default_admin_if_not_exists()

        print(f"[{datetime.utcnow()}] UserService initialized by user: xthanh1910")

    def _create_default_admin_if_not_exists(self):
        """
        Tạo admin user mặc định nếu chưa có (cho development)
        """
        try:
            admin_user_id = "admin_xthanh1910"

            # Kiểm tra admin đã tồn tại chưa
            if admin_user_id not in self._mock_users:
                # Tạo admin user
                admin_data = {
                    'user_id': admin_user_id,
                    'username': 'xthanh1910',
                    'email': 'xthanh1910@admin.com',
                    'full_name': 'Admin User',
                    'role': 'admin',
                    'status': 'active',
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow(),
                    'is_verified': True,
                    'last_login': None
                }

                # Hash password mặc định
                password_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                auth_data = {
                    'user_id': admin_user_id,
                    'password_hash': password_hash,
                    'login_attempts': 0,
                    'locked_until': None,
                    'last_password_change': datetime.utcnow()
                }

                # Profile mặc định
                profile_data = {
                    'user_id': admin_user_id,
                    'avatar_url': None,
                    'bio': 'System Administrator',
                    'preferences': {
                        'theme': 'dark',
                        'language': 'vi',
                        'notifications': True
                    },
                    'settings': {
                        'privacy_level': 'private',
                        'allow_contact': False
                    }
                }

                # Lưu vào mock storage
                self._mock_users[admin_user_id] = admin_data
                self._mock_auth[admin_user_id] = auth_data
                self._mock_profiles[admin_user_id] = profile_data
                self._mock_activities[admin_user_id] = []

                print(f"[{datetime.utcnow()}] Default admin user created: xthanh1910")

        except Exception as e:
            print(f"Error creating default admin: {str(e)}")

    # ================================
    # USER AUTHENTICATION METHODS
    # ================================

    async def authenticate_user(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """
        Xác thực user login
        Dev2: Kiểm tra credentials → tạo session → trả về auth data
        """
        try:
            # Bước 1: Tìm user theo username hoặc email
            user_data = await self._find_user_by_username_or_email(username_or_email)
            if not user_data:
                return {
                    'success': False,
                    'error': 'User không tồn tại',
                    'error_code': 'USER_NOT_FOUND'
                }

            user_id = user_data['user_id']

            # Bước 2: Kiểm tra account có bị khóa không
            auth_data = await self._get_user_auth_data(user_id)
            if not auth_data:
                return {
                    'success': False,
                    'error': 'Dữ liệu xác thực không tồn tại',
                    'error_code': 'AUTH_DATA_MISSING'
                }

            # Kiểm tra account lockout
            if self._is_account_locked(auth_data):
                return {
                    'success': False,
                    'error': 'Tài khoản tạm thời bị khóa do đăng nhập sai quá nhiều lần',
                    'error_code': 'ACCOUNT_LOCKED',
                    'locked_until': auth_data.get('locked_until')
                }

            # Bước 3: Kiểm tra user status
            if user_data.get('status') != 'active':
                return {
                    'success': False,
                    'error': f"Tài khoản {user_data.get('status', 'inactive')}",
                    'error_code': 'ACCOUNT_INACTIVE'
                }

            # Bước 4: Verify password
            password_hash = auth_data.get('password_hash')
            if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                # Tăng login attempts
                await self._increment_login_attempts(user_id)

                return {
                    'success': False,
                    'error': 'Mật khẩu không chính xác',
                    'error_code': 'INVALID_PASSWORD'
                }

            # Bước 5: Login thành công - reset login attempts
            await self._reset_login_attempts(user_id)

            # Bước 6: Tạo JWT token
            token_data = {
                'user_id': user_id,
                'username': user_data['username'],
                'role': user_data.get('role', 'user'),
                'exp': datetime.utcnow() + timedelta(hours=self.jwt_expire_hours),
                'iat': datetime.utcnow()
            }

            access_token = jwt.encode(token_data, self.jwt_secret, algorithm=self.jwt_algorithm)

            # Bước 7: Tạo user session
            session_data = {
                'session_id': str(uuid.uuid4()),
                'user_id': user_id,
                'access_token': access_token,
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(hours=self.jwt_expire_hours),
                'ip_address': None,  # Sẽ được set từ request
                'user_agent': None,  # Sẽ được set từ request
                'is_active': True
            }

            # Lưu session vào cache và database
            await self._save_user_session(session_data)

            # Bước 8: Cập nhật last_login
            await self._update_last_login(user_id)

            # Bước 9: Log login activity
            await self._log_user_activity(
                user_id=user_id,
                activity_type='login',
                activity_data={
                    'method': 'password',
                    'session_id': session_data['session_id']
                }
            )

            # Bước 10: Trả về authentication response
            return {
                'success': True,
                'access_token': access_token,
                'token_type': 'Bearer',
                'expires_in': self.jwt_expire_hours * 3600,  # seconds
                'user': {
                    'user_id': user_id,
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'full_name': user_data.get('full_name'),
                    'role': user_data.get('role', 'user'),
                    'avatar_url': await self._get_user_avatar(user_id)
                },
                'session_id': session_data['session_id']
            }

        except Exception as e:
            print(f"Error in authenticate_user: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi hệ thống trong quá trình xác thực',
                'error_code': 'SYSTEM_ERROR'
            }

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tạo user mới
        """
        try:
            # Bước 1: Validate dữ liệu đầu vào
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if field not in user_data or not user_data[field]:
                    return {
                        'success': False,
                        'error': f'Thiếu thông tin bắt buộc: {field}',
                        'error_code': 'MISSING_REQUIRED_FIELD'
                    }

            username = user_data['username'].strip()
            email = user_data['email'].strip().lower()
            password = user_data['password']

            # Validate username
            if len(username) < 3 or len(username) > 50:
                return {
                    'success': False,
                    'error': 'Username phải từ 3-50 ký tự',
                    'error_code': 'INVALID_USERNAME_LENGTH'
                }

            # Validate email format (basic)
            if '@' not in email or '.' not in email:
                return {
                    'success': False,
                    'error': 'Email không hợp lệ',
                    'error_code': 'INVALID_EMAIL_FORMAT'
                }

            # Validate password
            if len(password) < self.password_min_length:
                return {
                    'success': False,
                    'error': f'Mật khẩu phải có ít nhất {self.password_min_length} ký tự',
                    'error_code': 'PASSWORD_TOO_SHORT'
                }

            # Bước 2: Kiểm tra username và email đã tồn tại chưa
            existing_user = await self._find_user_by_username_or_email(username)
            if existing_user:
                return {
                    'success': False,
                    'error': 'Username đã tồn tại',
                    'error_code': 'USERNAME_EXISTS'
                }

            existing_email = await self._find_user_by_username_or_email(email)
            if existing_email:
                return {
                    'success': False,
                    'error': 'Email đã được sử dụng',
                    'error_code': 'EMAIL_EXISTS'
                }

            # Bước 3: Tạo user ID và hash password
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Bước 4: Tạo user data
            new_user = {
                'user_id': user_id,
                'username': username,
                'email': email,
                'full_name': user_data.get('full_name', ''),
                'role': 'user',  # Default role
                'status': 'active',
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'is_verified': False,  # Sẽ verify qua email
                'last_login': None
            }

            # Bước 5: Tạo auth data
            auth_data = {
                'user_id': user_id,
                'password_hash': password_hash,
                'login_attempts': 0,
                'locked_until': None,
                'last_password_change': datetime.utcnow()
            }

            # Bước 6: Tạo profile data
            profile_data = {
                'user_id': user_id,
                'avatar_url': None,
                'bio': '',
                'preferences': {
                    'theme': 'light',
                    'language': 'vi',
                    'notifications': True
                },
                'settings': {
                    'privacy_level': 'public',
                    'allow_contact': True
                }
            }

            # Bước 7: Lưu vào database (Dev3)
            if self.user_db:
                await self.user_db.create_user(new_user)
                await self.auth_db.create_user_auth(auth_data)
                await self.user_db.create_user_profile(profile_data)
            else:
                # Mock storage
                self._mock_users[user_id] = new_user
                self._mock_auth[user_id] = auth_data
                self._mock_profiles[user_id] = profile_data
                self._mock_activities[user_id] = []

            # Bước 8: Log user creation activity
            await self._log_user_activity(
                user_id=user_id,
                activity_type='user_created',
                activity_data={
                    'username': username,
                    'email': email,
                    'created_by': 'self_registration'
                }
            )

            # Bước 9: Trả về thông tin user (không bao gồm password)
            return {
                'success': True,
                'user': {
                    'user_id': user_id,
                    'username': username,
                    'email': email,
                    'full_name': new_user['full_name'],
                    'role': new_user['role'],
                    'status': new_user['status'],
                    'created_at': new_user['created_at'].isoformat()
                },
                'message': 'Tài khoản đã được tạo thành công'
            }

        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi hệ thống khi tạo tài khoản',
                'error_code': 'SYSTEM_ERROR'
            }

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token và lấy thông tin user
        """
        try:
            # Bước 1: Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get('user_id')

            if not user_id:
                return {
                    'valid': False,
                    'error': 'Token không hợp lệ',
                    'error_code': 'INVALID_TOKEN'
                }

            # Bước 2: Kiểm tra user còn tồn tại và active không
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {
                    'valid': False,
                    'error': 'User không tồn tại',
                    'error_code': 'USER_NOT_FOUND'
                }

            if user_data.get('status') != 'active':
                return {
                    'valid': False,
                    'error': 'Tài khoản không hoạt động',
                    'error_code': 'ACCOUNT_INACTIVE'
                }

            # Bước 3: Kiểm tra session còn hợp lệ không
            session_valid = await self._is_session_valid(token)
            if not session_valid:
                return {
                    'valid': False,
                    'error': 'Session đã hết hạn',
                    'error_code': 'SESSION_EXPIRED'
                }

            # Bước 4: Trả về user info
            return {
                'valid': True,
                'user': {
                    'user_id': user_id,
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'full_name': user_data.get('full_name'),
                    'role': user_data.get('role', 'user'),
                    'is_admin': user_data.get('role') == 'admin'
                },
                'token_data': payload
            }

        except jwt.ExpiredSignatureError:
            return {
                'valid': False,
                'error': 'Token đã hết hạn',
                'error_code': 'TOKEN_EXPIRED'
            }
        except jwt.InvalidTokenError:
            return {
                'valid': False,
                'error': 'Token không hợp lệ',
                'error_code': 'INVALID_TOKEN'
            }
        except Exception as e:
            print(f"Error verifying token: {str(e)}")
            return {
                'valid': False,
                'error': 'Lỗi hệ thống khi verify token',
                'error_code': 'SYSTEM_ERROR'
            }

    async def logout_user(self, token: str) -> Dict[str, Any]:
        """
        Đăng xuất user (invalidate token/session)
        """
        try:
            # Bước 1: Decode token để lấy user info
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
                user_id = payload.get('user_id')
            except jwt.InvalidTokenError:
                return {
                    'success': False,
                    'error': 'Token không hợp lệ',
                    'error_code': 'INVALID_TOKEN'
                }

            # Bước 2: Invalidate session
            await self._invalidate_session(token)

            # Bước 3: Log logout activity
            if user_id:
                await self._log_user_activity(
                    user_id=user_id,
                    activity_type='logout',
                    activity_data={'method': 'manual'}
                )

            return {
                'success': True,
                'message': 'Đăng xuất thành công'
            }

        except Exception as e:
            print(f"Error in logout: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi hệ thống khi đăng xuất',
                'error_code': 'SYSTEM_ERROR'
            }

    # ================================
    # USER MANAGEMENT METHODS
    # ================================

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin profile của user
        """
        try:
            # Lấy user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return None

            # Lấy profile data
            profile_data = await self._get_user_profile_data(user_id)

            # Combine user + profile data
            combined_profile = {
                **user_data,
                'profile': profile_data,
                'last_active': await self._get_last_activity_time(user_id)
            }

            # Ẩn sensitive data
            if 'password_hash' in combined_profile:
                del combined_profile['password_hash']

            return combined_profile

        except Exception as e:
            print(f"Error getting user profile: {str(e)}")
            return None

    async def update_user_profile(
        self,
        user_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cập nhật profile user
        """
        try:
            # Bước 1: Kiểm tra user tồn tại
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {
                    'success': False,
                    'error': 'User không tồn tại',
                    'error_code': 'USER_NOT_FOUND'
                }

            # Bước 2: Validate update data
            allowed_user_fields = ['full_name', 'email']
            allowed_profile_fields = ['bio', 'avatar_url', 'preferences', 'settings']

            user_updates = {}
            profile_updates = {}

            for field, value in update_data.items():
                if field in allowed_user_fields:
                    user_updates[field] = value
                elif field in allowed_profile_fields:
                    profile_updates[field] = value

            # Bước 3: Validate email nếu được update
            if 'email' in user_updates:
                new_email = user_updates['email'].strip().lower()
                if '@' not in new_email or '.' not in new_email:
                    return {
                        'success': False,
                        'error': 'Email không hợp lệ',
                        'error_code': 'INVALID_EMAIL'
                    }

                # Kiểm tra email đã được sử dụng chưa
                existing_user = await self._find_user_by_username_or_email(new_email)
                if existing_user and existing_user['user_id'] != user_id:
                    return {
                        'success': False,
                        'error': 'Email đã được sử dụng',
                        'error_code': 'EMAIL_EXISTS'
                    }

                user_updates['email'] = new_email
                user_updates['is_verified'] = False  # Cần verify lại email mới

            # Bước 4: Update user data
            if user_updates:
                user_updates['updated_at'] = datetime.utcnow()

                if self.user_db:
                    await self.user_db.update_user(user_id, user_updates)
                else:
                    # Mock storage
                    if user_id in self._mock_users:
                        self._mock_users[user_id].update(user_updates)

            # Bước 5: Update profile data
            if profile_updates:
                if self.user_db:
                    await self.user_db.update_user_profile(user_id, profile_updates)
                else:
                    # Mock storage
                    if user_id in self._mock_profiles:
                        self._mock_profiles[user_id].update(profile_updates)

            # Bước 6: Log update activity
            await self._log_user_activity(
                user_id=user_id,
                activity_type='profile_updated',
                activity_data={
                    'updated_fields': list(user_updates.keys()) + list(profile_updates.keys())
                }
            )

            # Bước 7: Trả về updated profile
            updated_profile = await self.get_user_profile(user_id)

            return {
                'success': True,
                'user': updated_profile,
                'message': 'Cập nhật profile thành công'
            }

        except Exception as e:
            print(f"Error updating user profile: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi hệ thống khi cập nhật profile',
                'error_code': 'SYSTEM_ERROR'
            }

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> Dict[str, Any]:
        """
        Đổi mật khẩu user
        """
        try:
            # Bước 1: Lấy auth data để verify current password
            auth_data = await self._get_user_auth_data(user_id)
            if not auth_data:
                return {
                    'success': False,
                    'error': 'Dữ liệu xác thực không tồn tại',
                    'error_code': 'AUTH_DATA_MISSING'
                }

            # Bước 2: Verify current password
            current_password_hash = auth_data.get('password_hash')
            if not bcrypt.checkpw(current_password.encode('utf-8'), current_password_hash.encode('utf-8')):
                return {
                    'success': False,
                    'error': 'Mật khẩu hiện tại không chính xác',
                    'error_code': 'INVALID_CURRENT_PASSWORD'
                }

            # Bước 3: Validate new password
            if len(new_password) < self.password_min_length:
                return {
                    'success': False,
                    'error': f'Mật khẩu mới phải có ít nhất {self.password_min_length} ký tự',
                    'error_code': 'PASSWORD_TOO_SHORT'
                }

            # Bước 4: Hash new password
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Bước 5: Update password trong database
            auth_updates = {
                'password_hash': new_password_hash,
                'last_password_change': datetime.utcnow(),
                'login_attempts': 0,  # Reset login attempts
                'locked_until': None   # Remove any lockout
            }

            if self.auth_db:
                await self.auth_db.update_user_auth(user_id, auth_updates)
            else:
                # Mock storage
                if user_id in self._mock_auth:
                    self._mock_auth[user_id].update(auth_updates)

            # Bước 6: Invalidate tất cả sessions hiện tại (bắt buộc login lại)
            await self._invalidate_all_user_sessions(user_id)

            # Bước 7: Log password change activity
            await self._log_user_activity(
                user_id=user_id,
                activity_type='password_changed',
                activity_data={'changed_at': datetime.utcnow().isoformat()}
            )

            return {
                'success': True,
                'message': 'Đổi mật khẩu thành công. Vui lòng đăng nhập lại.'
            }

        except Exception as e:
            print(f"Error changing password: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi hệ thống khi đổi mật khẩu',
                'error_code': 'SYSTEM_ERROR'
            }

    # ================================
    # USER ACTIVITY TRACKING METHODS
    # ================================

    async def track_user_activity(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
        ip_address: str = None,
        user_agent: str = None
    ):
        """
        Track user activity cho analytics và security
        """
        try:
            activity_record = {
                'activity_id': str(uuid.uuid4()),
                'user_id': user_id,
                'activity_type': activity_type,
                'activity_data': activity_data,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'timestamp': datetime.utcnow()
            }

            # Lưu vào database (Dev3)
            if self.activity_db:
                await self.activity_db.log_user_activity(activity_record)
            else:
                # Mock storage
                if user_id not in self._mock_activities:
                    self._mock_activities[user_id] = []
                self._mock_activities[user_id].append(activity_record)

                # Giới hạn số lượng activities để tránh memory leak
                if len(self._mock_activities[user_id]) > 1000:
                    self._mock_activities[user_id] = self._mock_activities[user_id][-500:]

        except Exception as e:
            print(f"Error tracking user activity: {str(e)}")

    async def get_user_activities(
        self,
        user_id: str,
        limit: int = 50,
        activity_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử hoạt động của user
        """
        try:
            if self.activity_db:
                activities = await self.activity_db.get_user_activities(
                    user_id=user_id,
                    limit=limit,
                    activity_type=activity_type
                )
            else:
                # Mock storage
                user_activities = self._mock_activities.get(user_id, [])

                # Filter by type nếu có
                if activity_type:
                    user_activities = [
                        activity for activity in user_activities
                        if activity.get('activity_type') == activity_type
                    ]

                # Sort by timestamp desc và limit
                user_activities.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                activities = user_activities[:limit]

            # Format activities cho frontend
            formatted_activities = []
            for activity in activities:
                formatted_activities.append({
                    'activity_type': activity.get('activity_type'),
                    'activity_data': activity.get('activity_data', {}),
                    'timestamp': activity.get('timestamp'),
                    'ip_address': activity.get('ip_address'),
                    'user_agent': activity.get('user_agent')
                })

            return formatted_activities

        except Exception as e:
            print(f"Error getting user activities: {str(e)}")
            return []

    # ================================
    # ADMIN USER MANAGEMENT METHODS
    # ================================

    async def get_users_for_admin(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lấy danh sách users cho admin với filters và pagination
        """
        try:
            page = filters.get('page', 1)
            limit = filters.get('limit', 20)
            status_filter = filters.get('status')
            search_query = filters.get('search')

            if self.user_db:
                # Lấy từ database (Dev3)
                result = await self.user_db.get_users_for_admin(filters)
            else:
                # Mock storage
                all_users = list(self._mock_users.values())

                # Apply filters
                if status_filter:
                    all_users = [u for u in all_users if u.get('status') == status_filter]

                if search_query:
                    search_lower = search_query.lower()
                    all_users = [
                        u for u in all_users
                        if (search_lower in u.get('username', '').lower() or
                            search_lower in u.get('email', '').lower() or
                            search_lower in u.get('full_name', '').lower())
                    ]

                # Sort by created_at desc
                all_users.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)

                # Pagination
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                paginated_users = all_users[start_idx:end_idx]

                # Remove sensitive data
                for user in paginated_users:
                    if 'password_hash' in user:
                        del user['password_hash']

                result = {
                    'users': paginated_users,
                    'total': len(all_users),
                    'page': page,
                    'limit': limit,
                    'total_pages': (len(all_users) + limit - 1) // limit
                }

            return result

        except Exception as e:
            print(f"Error getting users for admin: {str(e)}")
            return {'users': [], 'total': 0, 'page': page, 'limit': limit}

    async def get_user_detail_for_admin(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết user cho admin
        """
        try:
            # Lấy basic user data
            user_data = await self.get_user_profile(user_id)
            if not user_data:
                return None

            # Lấy thêm admin-specific data
            auth_data = await self._get_user_auth_data(user_id)
            recent_activities = await self.get_user_activities(user_id, limit=20)

            # Combine all data
            admin_detail = {
                **user_data,
                'auth_info': {
                    'login_attempts': auth_data.get('login_attempts', 0) if auth_data else 0,
                    'locked_until': auth_data.get('locked_until') if auth_data else None,
                    'last_password_change': auth_data.get('last_password_change') if auth_data else None
                },
                'recent_activities': recent_activities,
                'admin_view': True,
                'retrieved_at': datetime.utcnow()
            }

            return admin_detail

        except Exception as e:
            print(f"Error getting user detail for admin: {str(e)}")
            return None

    async def update_user_status(
        self,
        user_id: str,
        new_status: str,
        reason: str,
        updated_by: str
    ) -> bool:
        """
        Cập nhật status user (admin only)
        """
        try:
            # Validate status
            valid_statuses = ['active', 'inactive', 'banned', 'suspended']
            if new_status not in valid_statuses:
                return False

            # Lấy current user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return False

            old_status = user_data.get('status')

            # Update status
            user_updates = {
                'status': new_status,
                'updated_at': datetime.utcnow()
            }

            if self.user_db:
                await self.user_db.update_user(user_id, user_updates)
            else:
                # Mock storage
                if user_id in self._mock_users:
                    self._mock_users[user_id].update(user_updates)

            # Invalidate user sessions nếu bị ban/suspend
            if new_status in ['banned', 'suspended', 'inactive']:
                await self._invalidate_all_user_sessions(user_id)

            # Log admin action
            await self._log_user_activity(
                user_id=user_id,
                activity_type='status_updated_by_admin',
                activity_data={
                    'old_status': old_status,
                    'new_status': new_status,
                    'reason': reason,
                    'updated_by': updated_by
                }
            )

            return True

        except Exception as e:
            print(f"Error updating user status: {str(e)}")
            return False

    async def delete_user_completely(
        self,
        user_id: str,
        deleted_by: str
    ) -> bool:
        """
        Xóa user hoàn toàn khỏi hệ thống (admin only, nguy hiểm)
        """
        try:
            # Lấy user data để log
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return False

            # Invalidate tất cả sessions
            await self._invalidate_all_user_sessions(user_id)

            # Xóa từ tất cả tables
            if self.user_db:
                await self.user_db.delete_user_completely(user_id)
                await self.auth_db.delete_user_auth(user_id)
                await self.activity_db.delete_user_activities(user_id)
            else:
                # Mock storage
                if user_id in self._mock_users:
                    del self._mock_users[user_id]
                if user_id in self._mock_auth:
                    del self._mock_auth[user_id]
                if user_id in self._mock_profiles:
                    del self._mock_profiles[user_id]
                if user_id in self._mock_activities:
                    del self._mock_activities[user_id]

            # Log deletion (tới central admin log)
            await self._log_user_activity(
                user_id=deleted_by,  # Log under admin's account
                activity_type='user_deleted_by_admin',
                activity_data={
                    'deleted_user_id': user_id,
                    'deleted_username': user_data.get('username'),
                    'deleted_email': user_data.get('email'),
                    'deletion_reason': 'admin_action'
                }
            )

            print(f"[{datetime.utcnow()}] User deleted completely: {user_id} by admin: {deleted_by}")
            return True

        except Exception as e:
            print(f"Error deleting user completely: {str(e)}")
            return False

    # ================================
    # STATISTICS & ANALYTICS METHODS
    # ================================

    async def get_admin_user_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê users cho admin dashboard
        """
        try:
            if self.user_db:
                stats = await self.user_db.get_admin_user_statistics()
            else:
                # Mock statistics
                total_users = len(self._mock_users)
                active_users = len([u for u in self._mock_users.values() if u.get('status') == 'active'])

                # Calculate users created in last 7 days
                week_ago = datetime.utcnow() - timedelta(days=7)
                new_this_week = len([
                    u for u in self._mock_users.values()
                    if u.get('created_at', datetime.min) > week_ago
                ])

                # Calculate active today (mock)
                active_today = min(total_users, max(1, total_users // 3))  # Mock: 1/3 users active today

                stats = {
                    'total_users': total_users,
                    'active_users': active_users,
                    'inactive_users': total_users - active_users,
                    'new_this_week': new_this_week,
                    'active_today': active_today,
                    'banned_users': len([u for u in self._mock_users.values() if u.get('status') == 'banned']),
                    'admin_users': len([u for u in self._mock_users.values() if u.get('role') == 'admin'])
                }

            return stats

        except Exception as e:
            print(f"Error getting admin user stats: {str(e)}")
            return {}

    async def get_total_user_count(self) -> int:
        """
        Lấy tổng số users trong hệ thống
        """
        try:
            if self.user_db:
                return await self.user_db.get_total_user_count()
            else:
                return len(self._mock_users)
        except Exception as e:
            print(f"Error getting total user count: {str(e)}")
            return 0

    async def get_user_basic_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin cơ bản của user (cho references từ services khác)
        """
        try:
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return None

            return {
                'user_id': user_id,
                'username': user_data.get('username'),
                'full_name': user_data.get('full_name'),
                'role': user_data.get('role'),
                'status': user_data.get('status'),
                'avatar_url': await self._get_user_avatar(user_id)
            }

        except Exception as e:
            print(f"Error getting user basic info: {str(e)}")
            return None

    # ================================
    # HELPER METHODS
    # ================================

    async def _find_user_by_username_or_email(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Tìm user theo username hoặc email
        """
        try:
            identifier = identifier.strip().lower()

            if self.user_db:
                return await self.user_db.find_user_by_username_or_email(identifier)
            else:
                # Mock storage
                for user in self._mock_users.values():
                    if (user.get('username', '').lower() == identifier or
                        user.get('email', '').lower() == identifier):
                        return user
                return None

        except Exception as e:
            print(f"Error finding user: {str(e)}")
            return None

    async def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy user data theo ID
        """
        try:
            if self.user_db:
                return await self.user_db.get_user_by_id(user_id)
            else:
                return self._mock_users.get(user_id)
        except Exception as e:
            print(f"Error getting user by ID: {str(e)}")
            return None

    async def _get_user_auth_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy auth data của user
        """
        try:
            if self.auth_db:
                return await self.auth_db.get_user_auth_data(user_id)
            else:
                return self._mock_auth.get(user_id)
        except Exception as e:
            print(f"Error getting user auth data: {str(e)}")
            return None

    async def _get_user_profile_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy profile data của user
        """
        try:
            if self.user_db:
                return await self.user_db.get_user_profile_data(user_id)
            else:
                return self._mock_profiles.get(user_id, {})
        except Exception as e:
            print(f"Error getting user profile data: {str(e)}")
            return {}

    async def _get_user_avatar(self, user_id: str) -> Optional[str]:
        """
        Lấy avatar URL của user
        """
        try:
            profile_data = await self._get_user_profile_data(user_id)
            return profile_data.get('avatar_url') if profile_data else None
        except Exception:
            return None

    def _is_account_locked(self, auth_data: Dict[str, Any]) -> bool:
        """
        Kiểm tra account có bị khóa không
        """
        locked_until = auth_data.get('locked_until')
        if not locked_until:
            return False

        if isinstance(locked_until, str):
            locked_until = datetime.fromisoformat(locked_until.replace('Z', '+00:00'))

        return datetime.utcnow() < locked_until

    async def _increment_login_attempts(self, user_id: str):
        """
        Tăng số lần login attempts và lock account nếu cần
        """
        try:
            auth_data = await self._get_user_auth_data(user_id)
            if not auth_data:
                return

            login_attempts = auth_data.get('login_attempts', 0) + 1
            updates = {'login_attempts': login_attempts}

            # Lock account nếu quá số lần cho phép
            if login_attempts >= self.max_login_attempts:
                lockout_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
                updates['locked_until'] = lockout_until

            # Update database
            if self.auth_db:
                await self.auth_db.update_user_auth(user_id, updates)
            else:
                # Mock storage
                if user_id in self._mock_auth:
                    self._mock_auth[user_id].update(updates)

        except Exception as e:
            print(f"Error incrementing login attempts: {str(e)}")

    async def _reset_login_attempts(self, user_id: str):
        """
        Reset login attempts khi login thành công
        """
        try:
            updates = {
                'login_attempts': 0,
                'locked_until': None
            }

            if self.auth_db:
                await self.auth_db.update_user_auth(user_id, updates)
            else:
                # Mock storage
                if user_id in self._mock_auth:
                    self._mock_auth[user_id].update(updates)

        except Exception as e:
            print(f"Error resetting login attempts: {str(e)}")

    async def _save_user_session(self, session_data: Dict[str, Any]):
        """
        Lưu user session vào cache và database
        """
        try:
            session_id = session_data['session_id']
            access_token = session_data['access_token']

            # Lưu vào cache để access nhanh
            await self.cache_service.set(
                key=f"session:{access_token}",
                value=session_data,
                ttl=self.jwt_expire_hours * 3600
            )

            # Lưu vào database để persistent
            if self.auth_db:
                await self.auth_db.save_user_session(session_data)
            else:
                # Mock storage
                self._mock_sessions[access_token] = session_data

        except Exception as e:
            print(f"Error saving user session: {str(e)}")

    async def _is_session_valid(self, token: str) -> bool:
        """
        Kiểm tra session còn hợp lệ không
        """
        try:
            # Check cache trước
            session_data = await self.cache_service.get(f"session:{token}")
            if session_data:
                expires_at = session_data.get('expires_at')
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                return datetime.utcnow() < expires_at

            # Check database
            if self.auth_db:
                session_data = await self.auth_db.get_session_by_token(token)
                if session_data and session_data.get('is_active'):
                    expires_at = session_data.get('expires_at')
                    if isinstance(expires_at, str):
                        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    return datetime.utcnow() < expires_at
            else:
                # Mock storage
                session_data = self._mock_sessions.get(token)
                if session_data and session_data.get('is_active'):
                    expires_at = session_data.get('expires_at')
                    if isinstance(expires_at, str):
                        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    return datetime.utcnow() < expires_at

            return False

        except Exception as e:
            print(f"Error checking session validity: {str(e)}")
            return False

    async def _invalidate_session(self, token: str):
        """
        Invalidate một session cụ thể
        """
        try:
            # Remove từ cache
            await self.cache_service.delete(f"session:{token}")

            # Update database
            if self.auth_db:
                await self.auth_db.invalidate_session(token)
            else:
                # Mock storage
                if token in self._mock_sessions:
                    self._mock_sessions[token]['is_active'] = False

        except Exception as e:
            print(f"Error invalidating session: {str(e)}")

    async def _invalidate_all_user_sessions(self, user_id: str):
        """
        Invalidate tất cả sessions của user
        """
        try:
            if self.auth_db:
                await self.auth_db.invalidate_all_user_sessions(user_id)
            else:
                # Mock storage
                for token, session_data in self._mock_sessions.items():
                    if session_data.get('user_id') == user_id:
                        session_data['is_active'] = False
                        # Remove từ cache
                        await self.cache_service.delete(f"session:{token}")

        except Exception as e:
            print(f"Error invalidating all user sessions: {str(e)}")

    async def _update_last_login(self, user_id: str):
        """
        Cập nhật last_login time
        """
        try:
            updates = {'last_login': datetime.utcnow()}

            if self.user_db:
                await self.user_db.update_user(user_id, updates)
            else:
                # Mock storage
                if user_id in self._mock_users:
                    self._mock_users[user_id].update(updates)

        except Exception as e:
            print(f"Error updating last login: {str(e)}")

    async def _log_user_activity(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any]
    ):
        """
        Log user activity (wrapper cho track_user_activity)
        """
        await self.track_user_activity(
            user_id=user_id,
            activity_type=activity_type,
            activity_data=activity_data
        )

    async def _get_last_activity_time(self, user_id: str) -> Optional[datetime]:
        """
        Lấy thời gian activity cuối cùng của user
        """
        try:
            activities = await self.get_user_activities(user_id, limit=1)
            if activities:
                return activities[0].get('timestamp')
            return None
        except Exception:
            return None

    # ================================
    # SERVICE MANAGEMENT METHODS
    # ================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check cho UserService
        """
        try:
            start_time = time.time()

            # Test database connections (Dev3)
            db_connections = {
                'user_db': True,
                'auth_db': True,
                'activity_db': True
            }

            if self.user_db:
                try:
                    db_connections['user_db'] = await self.user_db.health_check()
                except Exception:
                    db_connections['user_db'] = False

            if self.auth_db:
                try:
                    db_connections['auth_db'] = await self.auth_db.health_check()
                except Exception:
                    db_connections['auth_db'] = False

            if self.activity_db:
                try:
                    db_connections['activity_db'] = await self.activity_db.health_check()
                except Exception:
                    db_connections['activity_db'] = False

            # Test cache connection
            cache_result = await self.cache_service.health_check()
            cache_ok = cache_result.get("status") == "healthy"

            response_time = (time.time() - start_time) * 1000
            all_db_ok = all(db_connections.values())
            overall_status = "healthy" if (all_db_ok and cache_ok) else "degraded"

            return {
                "status": overall_status,
                "service": "user_service",
                "response_time_ms": round(response_time, 2),
                "components": {
                    "user_database": "healthy" if db_connections['user_db'] else "error",
                    "auth_database": "healthy" if db_connections['auth_db'] else "error",
                    "activity_database": "healthy" if db_connections['activity_db'] else "error",
                    "cache": "healthy" if cache_ok else "error"
                },
                "metrics": {
                    "total_users": len(self._mock_users),
                    "active_sessions": len([s for s in self._mock_sessions.values() if s.get('is_active')]),
                    "jwt_expire_hours": self.jwt_expire_hours,
                    "max_login_attempts": self.max_login_attempts
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def initialize(self):
        """
        Khởi tạo UserService khi app startup
        """
        try:
            # Initialize database connections (Dev3)
            if self.user_db:
                await self.user_db.connect()
            if self.auth_db:
                await self.auth_db.connect()
            if self.activity_db:
                await self.activity_db.connect()

            # Initialize cache service
            await self.cache_service.initialize()

            # Create default admin if not exists
            self._create_default_admin_if_not_exists()

            print(f"[{datetime.utcnow()}] UserService initialized successfully")

        except Exception as e:
            print(f"[{datetime.utcnow()}] UserService initialization failed: {str(e)}")
            raise

    async def shutdown(self):
        """
        Graceful shutdown UserService
        """
        try:
            # Close database connections (Dev3)
            if self.user_db:
                await self.user_db.disconnect()
            if self.auth_db:
                await self.auth_db.disconnect()
            if self.activity_db:
                await self.activity_db.disconnect()

            # Close cache service
            await self.cache_service.close()

            print(f"[{datetime.utcnow()}] UserService shutdown completed")

        except Exception as e:
            print(f"[{datetime.utcnow()}] UserService shutdown error: {str(e)}")

# Export service instance
user_service = UserService()