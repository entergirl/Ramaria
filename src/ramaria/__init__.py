"""Ramaria - 珊瑚菌 - 本地运行的个人 AI 陪伴记忆系统。

以「记忆」为核心，支持分层记忆、混合RAG检索、知识图谱。

版本信息请参阅 pyproject.toml
"""

__author__ = "黎烧酒"
__license__ = "MIT"

# 统一导出日志和异常模块
from ramaria.logger import (
    get_logger,
    init_logging,
    LogContext,
    traced,
    log_exception,
    wrap_exception,
    get_trace_id,
    set_trace_id,
    clear_trace_id,
    generate_trace_id,
    is_debug,
    LogLevel,
    RamariaLogger,
)

from ramaria.exceptions import (
    # 错误码
    ErrorCode,
    
    # 异常基类
    RamariaException,
    
    # 系统异常
    SystemException,
    InitFailedError,
    ShutdownFailedError,
    InternalError,
    InvalidStateError,
    TimeoutError,
    
    # LLM 异常
    LLMException,
    LLMConnectionError,
    LLMTimeoutError,
    LLMResponseParseError,
    LLMInvalidResponseError,
    LLMAPIError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    
    # 数据库异常
    DatabaseException,
    DBConnectionError,
    DBQueryError,
    DBIntegrityError,
    DBMigrationError,
    DBCorruptedError,
    
    # 向量数据库异常
    VectorDBException,
    VectorDBInitError,
    VectorDBSearchError,
    EmbeddingModelError,
    
    # 网络异常
    NetworkException,
    NetworkConnectionError,
    NetworkTimeoutError,
    HTTPError,
    
    # 配置异常
    ConfigException,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigInvalidValueError,
    MissingRequiredConfigError,
    EnvNotSetError,
    
    # 记忆异常
    MemoryException,
    SummarizeError,
    MemoryMergeError,
    GraphBuildError,
    MemoryRetrieveError,
    MemoryPushError,
    MemoryConflictError,
    ContextTooLongError,
    
    # 工具异常
    ToolException,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolImportError,
    
    # 会话异常
    SessionException,
    SessionNotFoundError,
    SessionExpiredError,
    SessionCreateError,
    SessionStateError,
    
    # 适配器异常
    AdapterException,
    AdapterInitError,
    AdapterAuthError,
    AdapterSendError,
    AdapterReceiveError,
    
    # 安全异常
    SecurityException,
    UnauthorizedError,
    ForbiddenError,
    RateLimitedError,
    
    # 便捷函数
    raise_error,
)

__all__ = [
    # 日志
    'get_logger',
    'init_logging',
    'LogContext',
    'traced',
    'log_exception',
    'wrap_exception',
    'get_trace_id',
    'set_trace_id',
    'clear_trace_id',
    'generate_trace_id',
    'is_debug',
    'LogLevel',
    'RamariaLogger',
    # 异常
    'ErrorCode',
    'RamariaException',
    'SystemException',
    'InitFailedError',
    'ShutdownFailedError',
    'InternalError',
    'InvalidStateError',
    'TimeoutError',
    'LLMException',
    'LLMConnectionError',
    'LLMTimeoutError',
    'LLMResponseParseError',
    'LLMInvalidResponseError',
    'LLMAPIError',
    'LLMModelNotFoundError',
    'LLMQuotaExceededError',
    'DatabaseException',
    'DBConnectionError',
    'DBQueryError',
    'DBIntegrityError',
    'DBMigrationError',
    'DBCorruptedError',
    'VectorDBException',
    'VectorDBInitError',
    'VectorDBSearchError',
    'EmbeddingModelError',
    'NetworkException',
    'NetworkConnectionError',
    'NetworkTimeoutError',
    'HTTPError',
    'ConfigException',
    'ConfigFileNotFoundError',
    'ConfigParseError',
    'ConfigInvalidValueError',
    'MissingRequiredConfigError',
    'EnvNotSetError',
    'MemoryException',
    'SummarizeError',
    'MemoryMergeError',
    'GraphBuildError',
    'MemoryRetrieveError',
    'MemoryPushError',
    'MemoryConflictError',
    'ContextTooLongError',
    'ToolException',
    'ToolNotFoundError',
    'ToolExecutionError',
    'ToolTimeoutError',
    'ToolImportError',
    'SessionException',
    'SessionNotFoundError',
    'SessionExpiredError',
    'SessionCreateError',
    'SessionStateError',
    'AdapterException',
    'AdapterInitError',
    'AdapterAuthError',
    'AdapterSendError',
    'AdapterReceiveError',
    'SecurityException',
    'UnauthorizedError',
    'ForbiddenError',
    'RateLimitedError',
    'raise_error',
]
