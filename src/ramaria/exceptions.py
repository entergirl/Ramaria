# -*- coding: utf-8 -*-
"""
src/ramaria/exceptions.py - Ramaria 统一错误码和异常定义模块

错误码设计原则:
- 错误码格式: R{模块前缀}{序号}，共6位
- 模块前缀定义:
  - 00: 系统级错误
  - 01: LLM/AI 相关
  - 02: 数据库/存储
  - 03: 网络/通信
  - 04: 配置/环境
  - 05: 记忆/上下文
  - 06: 工具/插件
  - 07: 会话/用户
  - 08: 适配器/集成
  - 09: 安全/权限
"""

from typing import Optional, Any, Dict
from enum import Enum


# =============================================================================
# 错误码定义 (Error Codes)
# =============================================================================

class ErrorCode(Enum):
    """Ramaria 统一错误码枚举
    
    格式: R{模块前缀}{序号}，共6位
    
    模块前缀定义:
      - 00: 系统级错误
      - 01: LLM/AI 相关
      - 02: 数据库/存储
      - 03: 网络/通信
      - 04: 配置/环境
      - 05: 记忆/上下文
      - 06: 工具/插件
      - 07: 会话/用户
      - 08: 适配器/集成
      - 09: 安全/权限
    """
    
    # ========== 系统级错误 (R00xxxx) ==========
    R000001 = ("R000001", "SYS", "UNKNOWN", "未知错误")
    R000002 = ("R000002", "SYS", "INIT_FAILED", "系统初始化失败")
    R000003 = ("R000003", "SYS", "SHUTDOWN_FAILED", "系统关闭失败")
    R000004 = ("R000004", "SYS", "INTERNAL_ERROR", "内部错误")
    R000005 = ("R000005", "SYS", "NOT_IMPLEMENTED", "功能未实现")
    R000006 = ("R000006", "SYS", "INVALID_STATE", "无效状态")
    R000007 = ("R000007", "SYS", "TIMEOUT", "操作超时")
    R000008 = ("R000008", "SYS", "CANCELLED", "操作被取消")
    
    # ========== LLM/AI 相关 (R01xxxx) ==========
    R010001 = ("R010001", "LLM", "CONNECTION_FAILED", "无法连接到 LLM 服务")
    R010002 = ("R010002", "LLM", "TIMEOUT", "LLM 请求超时")
    R010003 = ("R010003", "LLM", "RESPONSE_PARSE_ERROR", "LLM 响应解析失败")
    R010004 = ("R010004", "LLM", "INVALID_RESPONSE", "LLM 返回无效响应")
    R010005 = ("R010005", "LLM", "API_ERROR", "LLM API 调用失败")
    R010006 = ("R010006", "LLM", "MODEL_NOT_FOUND", "指定的模型不存在")
    R010007 = ("R010007", "LLM", "QUOTA_EXCEEDED", "API 配额超限")
    R010008 = ("R010008", "LLM", "CONTENT_FILTERED", "内容被安全过滤")
    
    # ========== 数据库/存储 (R02xxxx) ==========
    R020001 = ("R020001", "DB", "CONNECTION_FAILED", "数据库连接失败")
    R020002 = ("R020002", "DB", "QUERY_FAILED", "数据库查询失败")
    R020003 = ("R020003", "DB", "INSERT_FAILED", "数据插入失败")
    R020004 = ("R020004", "DB", "UPDATE_FAILED", "数据更新失败")
    R020005 = ("R020005", "DB", "DELETE_FAILED", "数据删除失败")
    R020006 = ("R020006", "DB", "INTEGRITY_ERROR", "数据完整性错误")
    R020007 = ("R020007", "DB", "MIGRATION_FAILED", "数据库迁移失败")
    R020008 = ("R020008", "DB", "CORRUPTED", "数据库文件损坏")
    R020009 = ("R020009", "DB", "SCHEMA_MISMATCH", "数据库 schema 不匹配")
    
    # ========== 向量存储 (R02xxxx, 续) ==========
    R020101 = ("R020101", "VDB", "INIT_FAILED", "向量数据库初始化失败")
    R020102 = ("R020102", "VDB", "INSERT_FAILED", "向量插入失败")
    R020103 = ("R020103", "VDB", "SEARCH_FAILED", "向量搜索失败")
    R020104 = ("R020104", "VDB", "MODEL_NOT_FOUND", "嵌入模型不存在")
    R020105 = ("R020105", "VDB", "MODEL_LOAD_FAILED", "嵌入模型加载失败")
    
    # ========== 网络/通信 (R03xxxx) ==========
    R030001 = ("R030001", "NET", "CONNECTION_FAILED", "网络连接失败")
    R030002 = ("R030002", "NET", "TIMEOUT", "网络请求超时")
    R030003 = ("R030003", "NET", "HTTP_ERROR", "HTTP 请求错误")
    R030004 = ("R030004", "NET", "DNS_FAILED", "DNS 解析失败")
    R030005 = ("R030005", "NET", "SSL_ERROR", "SSL 证书错误")
    R030006 = ("R030006", "NET", "WEBSOCKET_ERROR", "WebSocket 连接错误")
    R030007 = ("R030007", "NET", "PROXY_ERROR", "代理服务器错误")
    
    # ========== 配置/环境 (R04xxxx) ==========
    R040001 = ("R040001", "CFG", "FILE_NOT_FOUND", "配置文件不存在")
    R040002 = ("R040002", "CFG", "PARSE_ERROR", "配置文件解析错误")
    R040003 = ("R040003", "CFG", "INVALID_VALUE", "配置项值无效")
    R040004 = ("R040004", "CFG", "MISSING_REQUIRED", "缺少必需配置项")
    R040005 = ("R040005", "CFG", "ENV_NOT_SET", "环境变量未设置")
    R040006 = ("R040006", "CFG", "PERMISSION_DENIED", "文件权限不足")
    
    # ========== 记忆/上下文 (R05xxxx) ==========
    R050001 = ("R050001", "MEM", "SUMMARIZE_FAILED", "摘要生成失败")
    R050002 = ("R050002", "MEM", "MERGE_FAILED", "记忆合并失败")
    R050003 = ("R050003", "MEM", "GRAPH_BUILD_FAILED", "知识图谱构建失败")
    R050004 = ("R050004", "MEM", "RETRIEVE_FAILED", "记忆检索失败")
    R050005 = ("R050005", "MEM", "PUSH_FAILED", "推送任务失败")
    R050006 = ("R050006", "MEM", "CONFLICT_DETECTED", "检测到记忆冲突")
    R050007 = ("R050007", "MEM", "CONTEXT_TOO_LONG", "上下文长度超限")
    R050008 = ("R050008", "MEM", "STORAGE_FULL", "记忆存储已满")
    
    # ========== 工具/插件 (R06xxxx) ==========
    R060001 = ("R060001", "TOOL", "NOT_FOUND", "工具不存在")
    R060002 = ("R060002", "TOOL", "EXECUTION_FAILED", "工具执行失败")
    R060003 = ("R060003", "TOOL", "TIMEOUT", "工具执行超时")
    R060004 = ("R060004", "TOOL", "INVALID_PARAMS", "工具参数无效")
    R060005 = ("R060005", "TOOL", "IMPORT_FAILED", "工具导入失败")
    R060006 = ("R060006", "TOOL", "REGISTER_FAILED", "工具注册失败")
    
    # ========== 会话/用户 (R07xxxx) ==========
    R070001 = ("R070001", "SESSION", "NOT_FOUND", "会话不存在")
    R070002 = ("R070002", "SESSION", "EXPIRED", "会话已过期")
    R070003 = ("R070003", "SESSION", "INVALID_ID", "无效的会话 ID")
    R070004 = ("R070004", "SESSION", "CREATE_FAILED", "会话创建失败")
    R070005 = ("R070005", "SESSION", "STATE_CORRUPTED", "会话状态损坏")
    
    # ========== 适配器/集成 (R08xxxx) ==========
    R080001 = ("R080001", "ADAPTER", "INIT_FAILED", "适配器初始化失败")
    R080002 = ("R080002", "ADAPTER", "AUTH_FAILED", "适配器认证失败")
    R080003 = ("R080003", "ADAPTER", "SEND_FAILED", "消息发送失败")
    R080004 = ("R080004", "ADAPTER", "RECEIVE_FAILED", "消息接收失败")
    R080005 = ("R080005", "ADAPTER", "HOOK_ERROR", "钩子执行错误")
    
    # ========== 安全/权限 (R09xxxx) ==========
    R090001 = ("R090001", "SEC", "UNAUTHORIZED", "未授权访问")
    R090002 = ("R090002", "SEC", "FORBIDDEN", "禁止访问")
    R090003 = ("R090003", "SEC", "RATE_LIMITED", "请求频率超限")
    R090004 = ("R090004", "SEC", "INVALID_TOKEN", "无效的认证令牌")
    R090005 = ("R090005", "SEC", "TOKEN_EXPIRED", "认证令牌已过期")
    
    def __init__(self, code: str, prefix: str, error_name: str, message: str):
        self.code = code
        self.prefix = prefix
        self.error_name = error_name  # 使用 error_name 避免与 Enum.name 冲突
        self.default_message = message
    
    @property
    def full_message(self) -> str:
        """返回完整错误消息: [错误码] 描述"""
        return f"[{self.code}] {self.default_message}"
    
    def format(self, detail: str = None) -> str:
        """格式化错误消息，可添加详细信息"""
        msg = self.full_message
        if detail:
            msg = f"{msg} — {detail}"
        return msg


# =============================================================================
# 错误码到异常类型的映射
# =============================================================================

ERROR_CODE_TO_EXCEPTION: Dict[ErrorCode, type] = {}


def register_error_code_mapping(error_code: ErrorCode, exception_type: type) -> None:
    """注册错误码到异常类型的映射"""
    ERROR_CODE_TO_EXCEPTION[error_code] = exception_type


# =============================================================================
# 自定义异常基类
# =============================================================================

class RamariaException(Exception):
    """Ramaria 所有自定义异常的基类"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.cause = cause
        self.context = context or {}
        self.trace_id = trace_id
    
    @property
    def code_string(self) -> str:
        """获取错误码字符串"""
        return self.error_code.code if self.error_code else "R000001"
    
    @property
    def prefix(self) -> str:
        """获取错误码前缀"""
        return self.error_code.prefix if self.error_code else "SYS"
    
    def __str__(self) -> str:
        parts = [f"[{self.code_string}] {self.message}"]
        if self.trace_id:
            parts.append(f"(trace_id={self.trace_id})")
        if self.cause:
            parts.append(f"← {type(self.cause).__name__}: {self.cause}")
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于序列化"""
        return {
            "message": self.message,
            "error_code": self.code_string,
            "prefix": self.prefix,
            "cause": str(self.cause) if self.cause else None,
            "context": self.context,
            "trace_id": self.trace_id
        }


# =============================================================================
# 系统级异常 (R00xxxx)
# =============================================================================

class SystemException(RamariaException):
    """系统级异常基类"""
    pass


class InitFailedError(SystemException):
    """系统初始化失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "系统初始化失败",
            ErrorCode.R000002,
            cause,
            **kwargs
        )


class ShutdownFailedError(SystemException):
    """系统关闭失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "系统关闭失败",
            ErrorCode.R000003,
            cause,
            **kwargs
        )


class InternalError(SystemException):
    """内部错误"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "内部错误",
            ErrorCode.R000004,
            cause,
            **kwargs
        )


class NotImplementedError(RamariaException):
    """功能未实现"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "功能未实现",
            ErrorCode.R000005,
            **kwargs
        )


class InvalidStateError(SystemException):
    """无效状态"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "无效状态",
            ErrorCode.R000006,
            **kwargs
        )


class TimeoutError(RamariaException):
    """操作超时"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "操作超时",
            ErrorCode.R000007,
            **kwargs
        )


# =============================================================================
# LLM/AI 异常 (R01xxxx)
# =============================================================================

class LLMException(RamariaException):
    """LLM 相关异常基类"""
    pass


class LLMConnectionError(LLMException):
    """无法连接到 LLM 服务"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "无法连接到 LLM 服务",
            ErrorCode.R010001,
            cause,
            **kwargs
        )


class LLMTimeoutError(LLMException):
    """LLM 请求超时"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "LLM 请求超时",
            ErrorCode.R010002,
            **kwargs
        )


class LLMResponseParseError(LLMException):
    """LLM 响应解析失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "LLM 响应解析失败",
            ErrorCode.R010003,
            cause,
            **kwargs
        )


class LLMInvalidResponseError(LLMException):
    """LLM 返回无效响应"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "LLM 返回无效响应",
            ErrorCode.R010004,
            **kwargs
        )


class LLMAPIError(LLMException):
    """LLM API 调用失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "LLM API 调用失败",
            ErrorCode.R010005,
            cause,
            **kwargs
        )


class LLMModelNotFoundError(LLMException):
    """指定的模型不存在"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "指定的模型不存在",
            ErrorCode.R010006,
            **kwargs
        )


class LLMQuotaExceededError(LLMException):
    """API 配额超限"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "API 配额超限",
            ErrorCode.R010007,
            **kwargs
        )


# =============================================================================
# 数据库/存储异常 (R02xxxx)
# =============================================================================

class DatabaseException(RamariaException):
    """数据库异常基类"""
    pass


class DBConnectionError(DatabaseException):
    """数据库连接失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "数据库连接失败",
            ErrorCode.R020001,
            cause,
            **kwargs
        )


class DBQueryError(DatabaseException):
    """数据库查询失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "数据库查询失败",
            ErrorCode.R020002,
            cause,
            **kwargs
        )


class DBIntegrityError(DatabaseException):
    """数据完整性错误"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "数据完整性错误",
            ErrorCode.R020006,
            cause,
            **kwargs
        )


class DBMigrationError(DatabaseException):
    """数据库迁移失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "数据库迁移失败",
            ErrorCode.R020007,
            cause,
            **kwargs
        )


class DBCorruptedError(DatabaseException):
    """数据库文件损坏"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "数据库文件损坏",
            ErrorCode.R020008,
            cause,
            **kwargs
        )


# 向量数据库异常
class VectorDBException(RamariaException):
    """向量数据库异常基类"""
    pass


class VectorDBInitError(VectorDBException):
    """向量数据库初始化失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "向量数据库初始化失败",
            ErrorCode.R020101,
            cause,
            **kwargs
        )


class VectorDBSearchError(VectorDBException):
    """向量搜索失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "向量搜索失败",
            ErrorCode.R020103,
            cause,
            **kwargs
        )


class EmbeddingModelError(VectorDBException):
    """嵌入模型错误"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "嵌入模型加载失败",
            ErrorCode.R020105,
            cause,
            **kwargs
        )


# =============================================================================
# 网络/通信异常 (R03xxxx)
# =============================================================================

class NetworkException(RamariaException):
    """网络异常基类"""
    pass


class NetworkConnectionError(NetworkException):
    """网络连接失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "网络连接失败",
            ErrorCode.R030001,
            cause,
            **kwargs
        )


class NetworkTimeoutError(NetworkException):
    """网络请求超时"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "网络请求超时",
            ErrorCode.R030002,
            **kwargs
        )


class HTTPError(NetworkException):
    """HTTP 请求错误"""
    def __init__(self, message: str = None, status_code: int = None, **kwargs):
        context = kwargs.get("context", {})
        if status_code:
            context["status_code"] = status_code
        kwargs["context"] = context
        super().__init__(
            message or f"HTTP 请求错误 (status={status_code})",
            ErrorCode.R030003,
            **kwargs
        )


# =============================================================================
# 配置/环境异常 (R04xxxx)
# =============================================================================

class ConfigException(RamariaException):
    """配置异常基类"""
    pass


class ConfigFileNotFoundError(ConfigException):
    """配置文件不存在"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "配置文件不存在",
            ErrorCode.R040001,
            **kwargs
        )


class ConfigParseError(ConfigException):
    """配置文件解析错误"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "配置文件解析错误",
            ErrorCode.R040002,
            cause,
            **kwargs
        )


class ConfigInvalidValueError(ConfigException):
    """配置项值无效"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "配置项值无效",
            ErrorCode.R040003,
            **kwargs
        )


class MissingRequiredConfigError(ConfigException):
    """缺少必需配置项"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "缺少必需配置项",
            ErrorCode.R040004,
            **kwargs
        )


class EnvNotSetError(ConfigException):
    """环境变量未设置"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "环境变量未设置",
            ErrorCode.R040005,
            **kwargs
        )


# =============================================================================
# 记忆/上下文异常 (R05xxxx)
# =============================================================================

class MemoryException(RamariaException):
    """记忆模块异常基类"""
    pass


class SummarizeError(MemoryException):
    """摘要生成失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "摘要生成失败",
            ErrorCode.R050001,
            cause,
            **kwargs
        )


class MemoryMergeError(MemoryException):
    """记忆合并失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "记忆合并失败",
            ErrorCode.R050002,
            cause,
            **kwargs
        )


class GraphBuildError(MemoryException):
    """知识图谱构建失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "知识图谱构建失败",
            ErrorCode.R050003,
            cause,
            **kwargs
        )


class MemoryRetrieveError(MemoryException):
    """记忆检索失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "记忆检索失败",
            ErrorCode.R050004,
            cause,
            **kwargs
        )


class MemoryPushError(MemoryException):
    """推送任务失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "推送任务失败",
            ErrorCode.R050005,
            cause,
            **kwargs
        )


class MemoryConflictError(MemoryException):
    """检测到记忆冲突"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "检测到记忆冲突",
            ErrorCode.R050006,
            **kwargs
        )


class ContextTooLongError(MemoryException):
    """上下文长度超限"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "上下文长度超限",
            ErrorCode.R050007,
            **kwargs
        )


# =============================================================================
# 工具/插件异常 (R06xxxx)
# =============================================================================

class ToolException(RamariaException):
    """工具模块异常基类"""
    pass


class ToolNotFoundError(ToolException):
    """工具不存在"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "工具不存在",
            ErrorCode.R060001,
            **kwargs
        )


class ToolExecutionError(ToolException):
    """工具执行失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "工具执行失败",
            ErrorCode.R060002,
            cause,
            **kwargs
        )


class ToolTimeoutError(ToolException):
    """工具执行超时"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "工具执行超时",
            ErrorCode.R000007,
            **kwargs
        )


class ToolImportError(ToolException):
    """工具导入失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "工具导入失败",
            ErrorCode.R060005,
            cause,
            **kwargs
        )


# =============================================================================
# 会话异常 (R07xxxx)
# =============================================================================

class SessionException(RamariaException):
    """会话异常基类"""
    pass


class SessionNotFoundError(SessionException):
    """会话不存在"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "会话不存在",
            ErrorCode.R070001,
            **kwargs
        )


class SessionExpiredError(SessionException):
    """会话已过期"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "会话已过期",
            ErrorCode.R070002,
            **kwargs
        )


class SessionCreateError(SessionException):
    """会话创建失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "会话创建失败",
            ErrorCode.R070004,
            cause,
            **kwargs
        )


class SessionStateError(SessionException):
    """会话状态错误"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "会话状态损坏",
            ErrorCode.R070005,
            **kwargs
        )


# =============================================================================
# 适配器异常 (R08xxxx)
# =============================================================================

class AdapterException(RamariaException):
    """适配器异常基类"""
    pass


class AdapterInitError(AdapterException):
    """适配器初始化失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "适配器初始化失败",
            ErrorCode.R080001,
            cause,
            **kwargs
        )


class AdapterAuthError(AdapterException):
    """适配器认证失败"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "适配器认证失败",
            ErrorCode.R080002,
            **kwargs
        )


class AdapterSendError(AdapterException):
    """消息发送失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "消息发送失败",
            ErrorCode.R080003,
            cause,
            **kwargs
        )


class AdapterReceiveError(AdapterException):
    """消息接收失败"""
    def __init__(self, message: str = None, cause: Exception = None, **kwargs):
        super().__init__(
            message or "消息接收失败",
            ErrorCode.R080004,
            cause,
            **kwargs
        )


# =============================================================================
# 安全异常 (R09xxxx)
# =============================================================================

class SecurityException(RamariaException):
    """安全异常基类"""
    pass


class UnauthorizedError(SecurityException):
    """未授权访问"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "未授权访问",
            ErrorCode.R090001,
            **kwargs
        )


class ForbiddenError(SecurityException):
    """禁止访问"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "禁止访问",
            ErrorCode.R090002,
            **kwargs
        )


class RateLimitedError(SecurityException):
    """请求频率超限"""
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            message or "请求频率超限",
            ErrorCode.R090003,
            **kwargs
        )


# =============================================================================
# 便捷异常创建函数
# =============================================================================

def raise_error(
    error_code: ErrorCode,
    detail: str = None,
    cause: Exception = None,
    context: Dict[str, Any] = None,
    trace_id: str = None
) -> None:
    """便捷的异常抛出函数"""
    exception_class = ERROR_CODE_TO_EXCEPTION.get(error_code, RamariaException)
    raise exception_class(
        message=error_code.format(detail),
        error_code=error_code,
        cause=cause,
        context=context,
        trace_id=trace_id
    )


# 注册错误码映射
register_error_code_mapping(ErrorCode.R000002, InitFailedError)
register_error_code_mapping(ErrorCode.R000003, ShutdownFailedError)
register_error_code_mapping(ErrorCode.R000004, InternalError)
register_error_code_mapping(ErrorCode.R000005, NotImplementedError)
register_error_code_mapping(ErrorCode.R000006, InvalidStateError)
register_error_code_mapping(ErrorCode.R000007, TimeoutError)
register_error_code_mapping(ErrorCode.R010001, LLMConnectionError)
register_error_code_mapping(ErrorCode.R010002, LLMTimeoutError)
register_error_code_mapping(ErrorCode.R010003, LLMResponseParseError)
register_error_code_mapping(ErrorCode.R010004, LLMInvalidResponseError)
register_error_code_mapping(ErrorCode.R010005, LLMAPIError)
register_error_code_mapping(ErrorCode.R010006, LLMModelNotFoundError)
register_error_code_mapping(ErrorCode.R010007, LLMQuotaExceededError)
register_error_code_mapping(ErrorCode.R020001, DBConnectionError)
register_error_code_mapping(ErrorCode.R020002, DBQueryError)
register_error_code_mapping(ErrorCode.R020006, DBIntegrityError)
register_error_code_mapping(ErrorCode.R020007, DBMigrationError)
register_error_code_mapping(ErrorCode.R020008, DBCorruptedError)
register_error_code_mapping(ErrorCode.R020101, VectorDBInitError)
register_error_code_mapping(ErrorCode.R020103, VectorDBSearchError)
register_error_code_mapping(ErrorCode.R020105, EmbeddingModelError)
register_error_code_mapping(ErrorCode.R030001, NetworkConnectionError)
register_error_code_mapping(ErrorCode.R030002, NetworkTimeoutError)
register_error_code_mapping(ErrorCode.R030003, HTTPError)
register_error_code_mapping(ErrorCode.R040001, ConfigFileNotFoundError)
register_error_code_mapping(ErrorCode.R040002, ConfigParseError)
register_error_code_mapping(ErrorCode.R040003, ConfigInvalidValueError)
register_error_code_mapping(ErrorCode.R040004, MissingRequiredConfigError)
register_error_code_mapping(ErrorCode.R040005, EnvNotSetError)
register_error_code_mapping(ErrorCode.R050001, SummarizeError)
register_error_code_mapping(ErrorCode.R050002, MemoryMergeError)
register_error_code_mapping(ErrorCode.R050003, GraphBuildError)
register_error_code_mapping(ErrorCode.R050004, MemoryRetrieveError)
register_error_code_mapping(ErrorCode.R050005, MemoryPushError)
register_error_code_mapping(ErrorCode.R050006, MemoryConflictError)
register_error_code_mapping(ErrorCode.R050007, ContextTooLongError)
register_error_code_mapping(ErrorCode.R060001, ToolNotFoundError)
register_error_code_mapping(ErrorCode.R060002, ToolExecutionError)
register_error_code_mapping(ErrorCode.R000007, ToolTimeoutError)
register_error_code_mapping(ErrorCode.R060005, ToolImportError)
register_error_code_mapping(ErrorCode.R070001, SessionNotFoundError)
register_error_code_mapping(ErrorCode.R070002, SessionExpiredError)
register_error_code_mapping(ErrorCode.R070004, SessionCreateError)
register_error_code_mapping(ErrorCode.R070005, SessionStateError)
register_error_code_mapping(ErrorCode.R080001, AdapterInitError)
register_error_code_mapping(ErrorCode.R080002, AdapterAuthError)
register_error_code_mapping(ErrorCode.R080003, AdapterSendError)
register_error_code_mapping(ErrorCode.R080004, AdapterReceiveError)
register_error_code_mapping(ErrorCode.R090001, UnauthorizedError)
register_error_code_mapping(ErrorCode.R090002, ForbiddenError)
register_error_code_mapping(ErrorCode.R090003, RateLimitedError)
