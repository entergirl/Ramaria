# -*- coding: utf-8 -*-
"""
src/ramaria/logger.py - Ramaria 统一日志管理模块

设计特点:
- 标准化的日志格式: [时间戳] [级别] [模块名] [追踪ID] 消息
- 支持日志追踪ID (trace_id) 贯穿整个请求生命周期
- 支持结构化日志上下文
- 日志分级策略: DEBUG < INFO < WARNING < ERROR < CRITICAL
- 双通道输出: 控制台 + 文件
- 自动日志轮转
"""

import logging
import logging.handlers
import sys
import os
import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps
from contextvars import ContextVar
from threading import Lock

# 导入异常模块（避免循环导入）
from .exceptions import RamariaException, ErrorCode

# =============================================================================
# 日志级别定义
# =============================================================================

class LogLevel:
    """日志级别常量"""
    DEBUG = logging.DEBUG      # 10 - 详细调试信息
    INFO = logging.INFO       # 20 - 一般信息
    WARNING = logging.WARNING # 30 - 警告信息
    ERROR = logging.ERROR     # 40 - 错误信息
    CRITICAL = logging.CRITICAL  # 50 - 严重错误

# 日志级别名称映射（用于显示）
LOG_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT"
}

# =============================================================================
# 追踪ID上下文变量
# =============================================================================

# 当前线程的追踪ID
# 使用 ContextVar 而非全局字典，确保每个异步任务/线程独立持有，
# 任务结束时自动回收，无内存泄漏风险
_trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


def get_trace_id() -> Optional[str]:
    """获取当前追踪ID"""
    return _trace_id_var.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """设置当前追踪ID，返回设置的追踪ID"""
    if trace_id is None:
        trace_id = generate_trace_id()
    _trace_id_var.set(trace_id)
    return trace_id


def clear_trace_id() -> None:
    """清除当前追踪ID"""
    _trace_id_var.set(None)


def generate_trace_id() -> str:
    """生成新的追踪ID"""
    return datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]


# =============================================================================
# 标准化日志格式化器
# =============================================================================

class StandardFormatter(logging.Formatter):
    """
    标准化的日志格式化器
    
    格式: [时间] [级别] [模块] [追踪ID] 消息
    示例: [2025-01-15 10:30:45] [INFO] [core.llm] [abc123] LLM请求已发送
    """
    
    def __init__(
        self,
        fmt: str = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(trace_id)s] %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        use_colors: bool = True
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        
        # ANSI 颜色代码
        self.COLORS = {
            'DEBUG': '\033[36m',      # 青色
            'INFO': '\033[32m',        # 绿色
            'WARN': '\033[33m',        # 黄色
            'WARNING': '\033[33m',     # 黄色
            'ERROR': '\033[31m',       # 红色
            'CRIT': '\033[35m',        # 紫色
            'CRITICAL': '\033[35m',    # 紫色
            'RESET': '\033[0m'         # 重置
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 添加追踪ID
        trace_id = get_trace_id()
        if trace_id:
            record.trace_id = trace_id
        else:
            record.trace_id = "-"
        
        # 存储原始级别名称
        original_levelname = record.levelname
        
        # 获取格式化后的消息
        if self.use_colors and sys.stdout.isatty():
            # 终端有颜色支持
            color = self.COLORS.get(original_levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{original_levelname}{self.COLORS['RESET']}"
        
        # 格式化异常堆栈
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)
        else:
            record.exc_text = ""
        
        return super().format(record)


class CompactFormatter(logging.Formatter):
    """
    紧凑型日志格式化器（用于文件输出）
    
    格式: [时间] [级别] [模块] [追踪ID] 消息
    """
    
    def __init__(
        self,
        fmt: str = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(trace_id)s] %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S"
    ):
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 添加追踪ID
        trace_id = get_trace_id()
        if trace_id:
            record.trace_id = trace_id
        else:
            record.trace_id = "-"
        
        # 格式化异常堆栈
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)
        else:
            record.exc_text = ""
        
        return super().format(record)


# =============================================================================
# 结构化日志接口
# =============================================================================

class StructuredLogger:
    """
    结构化日志记录器
    
    支持以字典形式记录结构化日志，便于日志分析系统处理
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def _format_struct(self, **kwargs) -> str:
        """格式化结构化数据"""
        # 安全序列化，处理不可序列化的对象
        def safe_serializer(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Exception):
                return f"{type(obj).__name__}: {obj}"
            return str(obj)
        
        struct_data = {k: safe_serializer(v) for k, v in kwargs.items()}
        return json.dumps(struct_data, ensure_ascii=False, separators=(',', ':'))
    
    def debug_struct(self, message: str, **kwargs) -> None:
        """记录结构化 DEBUG 日志"""
        self._logger.debug(f"{message} {self._format_struct(**kwargs)}")
    
    def info_struct(self, message: str, **kwargs) -> None:
        """记录结构化 INFO 日志"""
        self._logger.info(f"{message} {self._format_struct(**kwargs)}")
    
    def warning_struct(self, message: str, **kwargs) -> None:
        """记录结构化 WARNING 日志"""
        self._logger.warning(f"{message} {self._format_struct(**kwargs)}")
    
    def error_struct(self, message: str, **kwargs) -> None:
        """记录结构化 ERROR 日志"""
        self._logger.error(f"{message} {self._format_struct(**kwargs)}")
    
    def exception_struct(self, message: str, **kwargs) -> None:
        """记录结构化 ERROR 日志（含异常堆栈）"""
        self._logger.exception(f"{message} {self._format_struct(**kwargs)}")


# =============================================================================
# 增强型日志记录器
# =============================================================================

class RamariaLogger:
    """
    Ramaria 增强型日志记录器
    
    提供更友好的日志记录接口
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._struct = StructuredLogger(logger)
    
    @property
    def struct(self) -> StructuredLogger:
        """获取结构化日志记录器"""
        return self._struct
    
    def debug(self, message: str, **kwargs) -> None:
        """记录 DEBUG 日志"""
        if kwargs:
            self._logger.debug(f"{message} | {self._format_kwargs(kwargs)}")
        else:
            self._logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """记录 INFO 日志"""
        if kwargs:
            self._logger.info(f"{message} | {self._format_kwargs(kwargs)}")
        else:
            self._logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录 WARNING 日志"""
        if kwargs:
            self._logger.warning(f"{message} | {self._format_kwargs(kwargs)}")
        else:
            self._logger.warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """记录 ERROR 日志"""
        if kwargs:
            self._logger.error(f"{message} | {self._format_kwargs(kwargs)}")
        else:
            self._logger.error(message)
    
    def exception(self, message: str, **kwargs) -> None:
        """记录 ERROR 日志（含异常堆栈）"""
        if kwargs:
            self._logger.exception(f"{message} | {self._format_kwargs(kwargs)}")
        else:
            self._logger.exception(message)
    
    def critical(self, message: str, **kwargs) -> None:
        """记录 CRITICAL 日志"""
        if kwargs:
            self._logger.critical(f"{message} | {self._format_kwargs(kwargs)}")
        else:
            self._logger.critical(message)
    
    # 别名方法
    warn = warning
    warn_exc = exception
    
    def _format_kwargs(self, kwargs: Dict[str, Any]) -> str:
        """格式化关键字参数"""
        def safe_str(v):
            if isinstance(v, Exception):
                return f"{type(v).__name__}: {v}"
            return str(v)
        return " ".join(f"{k}={safe_str(v)}" for k, v in kwargs.items())


# =============================================================================
# 日志上下文管理器
# =============================================================================

class LogContext:
    """
    日志上下文管理器
    
    用法:
        with LogContext(trace_id="request-123"):
            logger.info("处理请求")
    """
    
    def __init__(
        self,
        trace_id: str = None,
        context: Dict[str, Any] = None,
        logger: RamariaLogger = None,
        level: int = None
    ):
        self.trace_id = trace_id or generate_trace_id()
        self.context = context or {}
        self.logger = logger
        self.level = level
        self._old_trace_id = None
    
    def __enter__(self) -> 'LogContext':
        self._old_trace_id = get_trace_id()
        set_trace_id(self.trace_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果发生异常，记录异常信息
        if exc_type is not None and self.logger:
            self.logger.exception(
                f"异常退出上下文",
                trace_id=self.trace_id,
                exception_type=exc_type.__name__,
                exception_message=str(exc_val),
                context=self.context
            )
        
        # 恢复之前的追踪ID
        if self._old_trace_id:
            set_trace_id(self._old_trace_id)
        else:
            clear_trace_id()
        
        return False  # 不抑制异常


# =============================================================================
# 追踪装饰器
# =============================================================================

def traced(
    trace_id_param: str = None,
    log_level: int = logging.INFO,
    log_args: bool = False,
    log_result: bool = False
):
    """
    函数追踪装饰器
    
    用法:
        @traced()
        def my_function(arg1, arg2):
            ...
    
    参数:
        trace_id_param: 从函数参数中获取追踪ID的参数名
        log_level: 日志级别
        log_args: 是否记录函数参数
        log_result: 是否记录返回值
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取或生成追踪ID
            if trace_id_param and trace_id_param in kwargs:
                trace_id = kwargs[trace_id_param]
            else:
                trace_id = generate_trace_id()
            
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            with LogContext(trace_id=trace_id):
                logger = get_logger(func.__module__)
                
                # 记录函数调用
                if log_args and (args or kwargs):
                    logger.log(
                        log_level,
                        f"[追踪开始] {func_name}",
                        args=args,
                        kwargs=kwargs
                    )
                else:
                    logger.log(log_level, f"[追踪开始] {func_name}")
                
                start_time = datetime.now()
                
                try:
                    result = func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    if log_result:
                        logger.log(
                            log_level,
                            f"[追踪完成] {func_name}",
                            duration=f"{duration:.3f}s",
                            result=str(result)[:200]  # 限制结果长度
                        )
                    else:
                        logger.log(
                            log_level,
                            f"[追踪完成] {func_name}",
                            duration=f"{duration:.3f}s"
                        )
                    
                    return result
                    
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.exception(
                        f"[追踪异常] {func_name}",
                        duration=f"{duration:.3f}s",
                        exception_type=type(e).__name__
                    )
                    raise
        
        return wrapper
    return decorator


# =============================================================================
# 异常日志工具
# =============================================================================

def log_exception(
    logger: RamariaLogger,
    error_code: ErrorCode = None,
    context: Dict[str, Any] = None,
    message: str = None,
    exc: Exception = None,
    level: int = logging.ERROR
) -> None:
    """
    标准化的异常日志记录
    
    用法:
        try:
            some_operation()
        except Exception as e:
            log_exception(
                logger,
                error_code=ErrorCode.R010001,
                context={"user_id": user_id},
                message="操作失败"
            )
    """
    trace_id = get_trace_id() or "-"
    
    log_data = {
        "trace_id": trace_id
    }
    
    if error_code:
        log_data["error_code"] = error_code.code
        log_data["error_prefix"] = error_code.prefix
        log_data["error_name"] = error_code.name
    
    if context:
        log_data.update(context)
    
    if exc:
        log_data["exception_type"] = type(exc).__name__
        log_data["exception_message"] = str(exc)
    
    msg = message or (error_code.format() if error_code else "发生异常")
    
    if exc:
        logger.log(level, f"{msg} | {json.dumps(log_data, ensure_ascii=False)}")
        logger.exception(f"异常详情 | {json.dumps(log_data, ensure_ascii=False)}")
    else:
        logger.log(level, f"{msg} | {json.dumps(log_data, ensure_ascii=False)}")


def wrap_exception(
    error_code: ErrorCode,
    context: Dict[str, Any] = None,
    message: str = None,
    reraise: bool = True
):
    """
    异常包装装饰器
    
    用法:
        @wrap_exception(ErrorCode.R010001, context={"operation": "api_call"})
        def call_api():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RamariaException:
                # 已经是 Ramaria 异常，直接重新抛出
                raise
            except Exception as e:
                logger = get_logger(func.__module__)
                log_exception(
                    logger,
                    error_code=error_code,
                    context=context,
                    message=message,
                    exc=e
                )
                
                if reraise:
                    raise RamariaException(
                        message=error_code.format(message),
                        error_code=error_code,
                        cause=e,
                        context=context,
                        trace_id=get_trace_id()
                    )
                return None
        return wrapper
    return decorator


# =============================================================================
# 日志管理器
# =============================================================================

class LogManager:
    """日志管理器"""
    
    def __init__(self):
        self._loggers: Dict[str, RamariaLogger] = {}
        self._lock = Lock()
        self._initialized = False
        self._debug_mode = False
        self._log_dir = Path("logs")
        self._log_file = "ramaria.log"
    
    def initialize(
        self,
        debug: bool = False,
        log_dir: str = None,
        log_file: str = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ) -> None:
        """
        初始化日志系统
        
        参数:
            debug: 是否开启调试模式
            log_dir: 日志目录
            log_file: 日志文件名
            console_level: 控制台日志级别
            file_level: 文件日志级别
        """
        with self._lock:
            if self._initialized:
                return
            
            self._debug_mode = debug
            if log_dir:
                self._log_dir = Path(log_dir)
            if log_file:
                self._log_file = log_file
            
            # 创建日志目录
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._log_dir / self._log_file
            
            # 获取根日志记录器
            root_logger = logging.getLogger("ramaria")
            root_logger.setLevel(logging.DEBUG)  # 接收所有级别
            
            # 清除现有处理器
            root_logger.handlers.clear()
            
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level if not debug else logging.DEBUG)
            console_handler.setFormatter(StandardFormatter())
            root_logger.addHandler(console_handler)
            
            # 文件处理器（带轮转）
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(CompactFormatter())
            root_logger.addHandler(file_handler)
            
            self._initialized = True
            
            # 记录初始化日志（不使用 get_logger，避免死锁）
            # 使用标准 logging 模块记录，手动格式化消息
            init_logger = logging.getLogger("ramaria.logger")
            init_msg = (
                f"日志系统已初始化 | "
                f"debug={debug}, "
                f"log_dir={self._log_dir}, "
                f"log_file={self._log_file}, "
                f"console_level={LOG_LEVEL_NAMES.get(console_level, str(console_level))}, "
                f"file_level={LOG_LEVEL_NAMES.get(file_level, str(file_level))}"
            )
            init_logger.info(init_msg)
    
    def get_logger(self, name: str) -> RamariaLogger:
        """获取日志记录器"""
        with self._lock:
            if name in self._loggers:
                return self._loggers[name]
            
            # 确保名称以 ramaria. 开头
            if not name.startswith("ramaria"):
                if name.startswith("src.ramaria"):
                    name = name.replace("src.ramaria", "ramaria", 1)
                else:
                    name = f"ramaria.{name}"
            
            # 获取标准 logger
            std_logger = logging.getLogger(name)
            
            # 包装为 RamariaLogger
            ramaria_logger = RamariaLogger(std_logger)
            self._loggers[name] = ramaria_logger
            
            return ramaria_logger
    
    @property
    def is_debug_mode(self) -> bool:
        """是否调试模式"""
        return self._debug_mode


# 全局日志管理器实例
_log_manager = LogManager()

# 为了向后兼容，保留旧的 API
_initialized_loggers: Dict[str, RamariaLogger] = {}
_log_dir = Path("logs")


def get_logger(name: str) -> RamariaLogger:
    """
    获取日志记录器（主要入口）
    
    用法:
        logger = get_logger(__name__)
        logger.info("这是一条信息日志")
        logger.error("这是一条错误日志", user_id="123")
        logger.struct.info("结构化日志", action="login", status="success")
    """
    return _log_manager.get_logger(name)


def init_logging(
    debug: bool = False,
    log_dir: str = "logs",
    log_file: str = "ramaria.log"
) -> None:
    """初始化日志系统（兼容旧 API）"""
    _log_manager.initialize(
        debug=debug,
        log_dir=log_dir,
        log_file=log_file
    )


def is_debug() -> bool:
    """检查是否调试模式"""
    return _log_manager.is_debug_mode


# =============================================================================
# 便捷的日志宏
# =============================================================================

# 预配置的模块级别快捷日志
class _ModuleLogger:
    """模块级快捷日志"""
    
    def __getattr__(self, name: str):
        # 延迟获取 logger
        def log_func(level: int, message: str, **kwargs):
            logger = get_logger(name)
            logger.log(level, message, **kwargs)
        
        return log_func


# 使用方式: _log.DEBUG("module.name", "message")
_log = _ModuleLogger()


# =============================================================================
# 日志级别检查便捷函数
# =============================================================================

def is_enabled_for(level: int, logger_name: str = "ramaria") -> bool:
    """检查日志级别是否启用"""
    logger = logging.getLogger(logger_name)
    return logger.isEnabledFor(level)


def debug_enabled(logger_name: str = "ramaria") -> bool:
    """DEBUG 级别是否启用"""
    return is_enabled_for(logging.DEBUG, logger_name)


def info_enabled(logger_name: str = "ramaria") -> bool:
    """INFO 级别是否启用"""
    return is_enabled_for(logging.INFO, logger_name)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 日志级别
    'LogLevel',
    'LOG_LEVEL_NAMES',
    
    # 追踪ID
    'get_trace_id',
    'set_trace_id',
    'clear_trace_id',
    'generate_trace_id',
    
    # 日志上下文
    'LogContext',
    'traced',
    
    # 异常日志工具
    'log_exception',
    'wrap_exception',
    
    # 日志格式化器
    'StandardFormatter',
    'CompactFormatter',
    
    # 结构化日志
    'StructuredLogger',
    
    # 增强型日志记录器
    'RamariaLogger',
    
    # 主要入口
    'get_logger',
    'init_logging',
    'is_debug',
    'is_enabled_for',
    'debug_enabled',
    'info_enabled',
    
    # 日志管理器
    'LogManager',
    
    # 快捷日志
    '_log',
    
    # 向后兼容
    'R000001',  # 导出错误码常量
]
