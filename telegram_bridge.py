"""
telegram_bridge.py — RAMARIA × Telegram Bot 桥接服务
====================================================

功能：
  · 接收 Telegram 私聊消息，转发给本地 RAMARIA /chat 接口
  · 将 RAMARIA 的回复发回 Telegram
  · 仅响应指定用户（白名单），防止被别人滥用

使用方法：
  1. 创建 Bot：找 @BotFather，发送 /newbot，拿到 BOT_TOKEN
  2. 获取你的 Telegram ID：找 @userinfobot，它会告诉你你的 ID
  3. 设置环境变量：
       set TELEGRAM_BOT_TOKEN=你的token
       set ALLOWED_USER_ID=你的Telegram数字ID
  4. 启动：python telegram_bridge.py

依赖：pip install python-telegram-bot requests
"""

import os
import logging
import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# =============================================================================
# 配置
# =============================================================================

# Telegram Bot Token（从 @BotFather 获取）
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# 允许使用此 Bot 的 Telegram 用户 ID（数字），其他人发消息会被忽略
# 获取方式：找 @userinfobot 发任意消息，它会回复你的 ID
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "0"))

# 本地 RAMARIA 服务地址
RAMARIA_API_URL = "http://localhost:8000"

# 日志配置
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bridge")


# =============================================================================
# 权限检查
# =============================================================================

def _is_allowed(user_id: int) -> bool:
    """检查消息发送者是否在白名单内。"""
    if ALLOWED_USER_ID == 0:
        logger.warning("ALLOWED_USER_ID 未设置，所有用户都能使用！")
        return True
    return user_id == ALLOWED_USER_ID


# =============================================================================
# Telegram 命令处理器
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start 命令 —— 欢迎消息"""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("抱歉，你不是授权用户。")
        return

    await update.message.reply_text(
        "烧酒，我是黎杋枫 🍄\n"
        "通过 Telegram 跟你聊天，跟在网页上一样。\n"
        "直接发消息就好啦~"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/status 命令 —— 查看连接状态"""
    try:
        resp = requests.get(f"{RAMARIA_API_URL}/router/status", timeout=5)
        data = resp.json()
        mode = data.get("mode", "unknown")
        api_enabled = data.get("api_enabled", False)

        status_text = (
            f"📡 连接状态：正常\n"
            f"🤖 当前模式：{'线上' if mode == 'pending' else '本地'}\n"
            f"🔧 线上API：{'开启' if api_enabled else '关闭'}"
        )
        await update.message.reply_text(status_text)

    except requests.exceptions.ConnectionError:
        await update.message.reply_text(
            "⚠️ 无法连接到本地 RAMARIA 服务。\n"
            "请确认 RAMARIA 已启动（python main.py）。"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ 状态查询失败：{e}")


# =============================================================================
# 消息转发核心逻辑
# =============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    接收用户消息 → 转发给 RAMARIA → 回复发回 Telegram。

    处理 RAMARIA 回复中的 || 分隔符：
      · RAMARIA 用 || 表示"连发多条消息"（对标微信连发）
      · 这里将 || 分隔的各条拆成独立的 Telegram 消息，还原连发体验
    """
    user_id = update.effective_user.id
    if not _is_allowed(user_id):
        logger.warning(f"未授权用户 {user_id} 尝试使用 Bot")
        return

    user_text = update.message.text
    if not user_text or not user_text.strip():
        return

    logger.info(f"收到消息：{user_text[:50]}...")

    # 发送"正在输入"状态（让用户知道在处理）
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    # 调用 RAMARIA /chat 接口
    try:
        resp = requests.post(
            f"{RAMARIA_API_URL}/chat",
            json={"content": user_text},
            timeout=120,  # 本地模型可能需要较长时间
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "")
        mode = data.get("mode", "local")

        logger.info(f"RAMARIA 回复（{mode}）：{reply[:80]}...")

    except requests.exceptions.ConnectionError:
        await update.message.reply_text(
            "嗯……好像连不上我的本体（本地服务未启动）\n"
            "你确认电脑上 RAMARIA 在运行吗？"
        )
        return
    except requests.exceptions.Timeout:
        await update.message.reply_text(
            "等了好久都没想好怎么说……可能是本地模型卡住了。"
        )
        return
    except Exception as e:
        logger.error(f"调用 RAMARIA 失败：{e}")
        await update.message.reply_text("出了点小问题，稍后再试试？")
        return

    # 处理回复内容
    if not reply or not reply.strip():
        await update.message.reply_text("……")
        return

    # 解析 RAMARIA 的 || 分隔符，拆成多条消息
    # RAMARIA 的格式："说的内容 【附加信息】|| 另一条 【附加信息】"
    # 用 || 分隔，每条独立发送
    parts = [p.strip() for p in reply.split("||") if p.strip()]

    for i, part in enumerate(parts):
        # 处理【附加信息】：提取出来，用斜体或单独行展示
        # Telegram 支持的格式：*粗体* _斜体_ `代码`
        text = part.strip()

        # 将 【附加信息】 转换为 Telegram 格式（用 _斜体_）
        # 不做复杂解析，直接保留原文，Telegram 端自然显示
        # （如果你想在 Telegram 里把附加信息用不同样式显示，可以在这里处理）

        await update.message.reply_text(text)

        # 多条消息之间稍微延迟，模拟连发效果
        if i < len(parts) - 1:
            import asyncio
            await asyncio.sleep(0.5)


# =============================================================================
# 错误处理
# =============================================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """全局错误处理。"""
    logger.error(f"Telegram Bot 异常：{context.error}", exc_info=context.error)


# =============================================================================
# 启动
# =============================================================================

def main():
    if not BOT_TOKEN:
        print("=" * 50)
        print("错误：TELEGRAM_BOT_TOKEN 未设置")
        print()
        print("设置方法：")
        print("  1. 在 Telegram 找 @BotFather")
        print("  2. 发送 /newbot，按提示创建 Bot")
        print("  3. 复制 Bot Token")
        print("  4. 设置环境变量：")
        print('     set TELEGRAM_BOT_TOKEN=你的token')
        print('     set ALLOWED_USER_ID=你的Telegram用户ID')
        print("=" * 50)
        return

    print("=" * 50)
    print("🍄 RAMARIA Telegram Bridge 启动中...")
    print(f"   RAMARIA 地址：{RAMARIA_API_URL}")
    print(f"   白名单用户ID：{ALLOWED_USER_ID or '未设置（所有人可用）'}")
    print("=" * 50)

    # 创建 Telegram Bot 应用
    app = Application.builder().token(BOT_TOKEN).build()

    # 注册处理器
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    # 注册全局错误处理
    app.add_error_handler(error_handler)

    # 启动轮询（默认用 long polling，不需要公网端口）
    print("Bot 已启动，等待消息...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
