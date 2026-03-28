"""
importer/qq_import_cli.py — QQ 聊天记录导入命令行工具
=====================================================================

用途：
    开发和调试阶段使用。支持指定文件路径，先 dry run 显示诊断报告，
    用户确认后写入数据库，写入完成后追问是否立即生成 L1 摘要。

运行方式：
    # 基本用法（交互式：先预览，再确认导入，再追问是否生成摘要）
    python -m importer.qq_import_cli --file path/to/export.json

    # 只做 dry run，不写入数据库
    python -m importer.qq_import_cli --file path/to/export.json --dry-run

    # 跳过导入确认直接写入
    python -m importer.qq_import_cli --file path/to/export.json --yes

    # 导入后自动开始生成摘要（不追问）
    python -m importer.qq_import_cli --file path/to/export.json --yes --generate-l1

    # 只生成摘要（不导入新文件，处理已有的待处理 session）
    python -m importer.qq_import_cli --generate-l1

    # 指定 session 切割阈值
    python -m importer.qq_import_cli --file export.json --gap 30
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 里
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =============================================================================
# 子命令：导入（解析 + 写入 L0）
# =============================================================================

def cmd_import(args) -> int:
    """
    执行导入流程：解析文件 → 诊断报告 → 确认 → 写入 L0。

    返回：
        待处理 session 数（0 表示无待处理或导入失败），供后续追问使用
    """
    from importer.qq_parser import parse_qq_export
    from importer.qq_importer import import_sessions_to_db

    # 读取 gap_minutes（优先命令行参数，其次读 config）
    if args.gap is not None:
        gap_minutes = args.gap
    else:
        try:
            from config import L1_IDLE_MINUTES
            gap_minutes = L1_IDLE_MINUTES
        except ImportError:
            gap_minutes = 10
            print("（提示）未找到 config.py，使用默认间隔 10 分钟")

    # ── 第一阶段：解析文件，生成诊断报告 ──
    print(f"\n正在解析文件：{args.file}")
    print(f"session 切割阈值：{gap_minutes} 分钟")

    try:
        result = parse_qq_export(args.file, gap_minutes=gap_minutes)
    except FileNotFoundError as e:
        print(f"\n❌  文件不存在：{e}")
        return 0
    except ValueError as e:
        print(f"\n❌  格式错误：{e}")
        return 0

    # 打印诊断报告
    result.report.print_summary()

    if args.dry_run:
        print("（dry run 模式，不写入数据库）")
        return 0

    # ── 确认写入 ──
    total_sessions = result.report.session_count
    total_msgs     = (
        result.report.success_text
        + result.report.success_image
        + result.report.success_reply
        + result.report.degraded_reply_fallback
        + result.report.degraded_forward
        + result.report.degraded_card
        + result.report.degraded_audio
        + result.report.degraded_video
    )

    if total_sessions == 0:
        print("没有可导入的 session（可能全部为重复消息），退出。")
        return 0

    if not args.yes:
        print(f"\n准备写入 {total_sessions} 个 session，共约 {total_msgs} 条消息。")
        answer = input("确认导入？(y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消。")
            return 0

    # ── 第二阶段：写入 L0 ──
    print(f"\n开始写入数据库（L0 层）…")

    try:
        stats = import_sessions_to_db(result.parsed_sessions)
    except Exception as e:
        print(f"\n❌  写入过程发生错误：{e}")
        import traceback
        traceback.print_exc()
        return 0

    # 打印写入结果
    print()
    print("=" * 56)
    print("  L0 导入完成")
    print("=" * 56)
    print(f"  写入 session 数 : {stats['sessions_written']}")
    print(f"  写入消息数      : {stats['messages_written']}")
    if stats['sessions_failed'] > 0:
        print(f"  失败 session 数 : {stats['sessions_failed']}")
        for detail in stats['failed_details']:
            print(f"    {detail}")
    print("=" * 56)

    return stats["sessions_written"]


# =============================================================================
# 子命令：生成 L1 摘要
# =============================================================================

def cmd_generate_l1(skip_confirm: bool = False) -> None:
    """
    对所有待处理 session 批量生成 L1 摘要（同步执行）。

    参数：
        skip_confirm — True 时跳过确认提示，直接开始
    """
    from importer.l1_batch import run_batch_cli, get_pending_count

    # 查询待处理数量
    pending_count = get_pending_count()

    if pending_count == 0:
        print("\n没有待处理的 session，所有历史 session 已生成摘要。")
        return

    if pending_count < 0:
        print("\n❌  查询待处理 session 失败，请检查数据库连接。")
        return

    print(f"\n共有 {pending_count} 个 session 待生成摘要。")

    if not skip_confirm:
        print("（提示：生成过程中如有实时对话，会自动暂停等待；按 Ctrl+C 可中止）")
        answer = input("开始生成？(y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消。可以之后用 --generate-l1 参数单独触发。")
            return

    # 同步执行批处理（命令行版不使用后台线程）
    run_batch_cli(session_manager=None)


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QQ 聊天记录导入工具（qq-chat-exporter v5 格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 完整流程（导入 + 追问是否生成摘要）
  python -m importer.qq_import_cli --file export.json

  # 只做预览
  python -m importer.qq_import_cli --file export.json --dry-run

  # 导入后自动生成摘要
  python -m importer.qq_import_cli --file export.json --yes --generate-l1

  # 只触发摘要生成（不导入新文件）
  python -m importer.qq_import_cli --generate-l1
        """,
    )

    parser.add_argument(
        "--file", "-f",
        default=None,
        help="QQ Chat Exporter 导出的 JSON 文件路径（不传时只触发摘要生成）",
    )
    parser.add_argument(
        "--gap", "-g",
        type=int,
        default=None,
        help="session 切割时间间隔（分钟），默认读取 config.py 的 L1_IDLE_MINUTES",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="只显示诊断报告，不写入数据库",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过导入确认提示，直接写入数据库",
    )
    parser.add_argument(
        "--generate-l1",
        action="store_true",
        help="导入完成后生成 L1 摘要（或单独触发摘要生成）",
    )

    args = parser.parse_args()

    # ── 没有传 --file，直接走"只生成摘要"逻辑 ──
    if args.file is None:
        if args.generate_l1:
            cmd_generate_l1(skip_confirm=False)
        else:
            parser.print_help()
        sys.exit(0)

    # ── 有 --file，先导入 ──
    sessions_written = cmd_import(args)

    # ── 导入成功后处理摘要生成逻辑 ──
    if args.dry_run or sessions_written == 0:
        sys.exit(0)

    if args.generate_l1:
        # 命令行参数明确指定了 --generate-l1，跳过追问直接开始
        cmd_generate_l1(skip_confirm=True)
    else:
        # 追问用户是否立即生成摘要
        print(f"\n导入完成。现在有若干历史 session 尚未生成摘要。")
        print("生成摘要后，历史对话内容才会真正融入记忆系统（L1/L2/L3）。")
        print("（提示：也可以之后用 --generate-l1 参数单独触发，或在网页界面操作）")
        answer = input("\n是否立即开始生成摘要？(y/n/稍后) [默认: 稍后]: ").strip().lower()
        if answer in ("y", "yes"):
            cmd_generate_l1(skip_confirm=True)
        else:
            print("好的，之后可以随时运行：")
            print("  python -m importer.qq_import_cli --generate-l1")
            print("  或在网页中打开"导入"页面操作。")


if __name__ == "__main__":
    main()
