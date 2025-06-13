from typing import Dict, Optional
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import logging
from pathlib import Path
from pydantic import BaseModel

class UserRole(BaseModel):
    role: str
    permissions: list[str]

class SecurityConfig:
    SECRET_KEY = "your-secret-key"  # Should be loaded from environment
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

class SecurityManager:
    def __init__(self, audit_log_dir: str = "logs/audit"):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logging
        logging.basicConfig(
            filename=self.audit_log_dir / "audit.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger("security_audit")
        
        # Role definitions
        self.roles = {
            "doctor": UserRole(
                role="doctor",
                permissions=["read", "write", "review"]
            ),
            "nurse": UserRole(
                role="nurse",
                permissions=["read", "write"]
            ),
            "reviewer": UserRole(
                role="reviewer",
                permissions=["read", "review"]
            )
        }
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            SecurityConfig.SECRET_KEY,
            algorithm=SecurityConfig.ALGORITHM
        )
        
        self.log_audit_event(
            "token_created",
            f"Access token created for user {data.get('sub')}",
            data.get('sub')
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.SECRET_KEY,
                algorithms=[SecurityConfig.ALGORITHM]
            )
            self.log_audit_event(
                "token_verified",
                f"Token verified for user {payload.get('sub')}",
                payload.get('sub')
            )
            return payload
        except jwt.JWTError:
            self.log_audit_event(
                "token_invalid",
                "Invalid token verification attempt",
                None
            )
            raise ValueError("Could not validate credentials")
    
    def verify_permission(self, user_role: str, required_permission: str) -> bool:
        """Verify if user has required permission"""
        if user_role not in self.roles:
            return False
        return required_permission in self.roles[user_role].permissions
    
    def log_audit_event(self, event_type: str, description: str, user_id: Optional[str] = None):
        """Log security audit event"""
        self.logger.info(
            f"EVENT: {event_type} - "
            f"USER: {user_id or 'anonymous'} - "
            f"DESC: {description}"
        )
    
    def get_audit_trail(self, user_id: Optional[str] = None) -> list[str]:
        """Get audit trail for a user"""
        audit_entries = []
        with open(self.audit_log_dir / "audit.log", "r") as f:
            for line in f:
                if not user_id or f"USER: {user_id}" in line:
                    audit_entries.append(line.strip())
        return audit_entries
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password) 