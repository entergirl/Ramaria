"""
conflict_checker.py — 记忆冲突检测模块
负责在 L1 摘要生成后，检测新内容是否与现有用户画像存在矛盾，
将发现的冲突写入 conflict_queue 表，并在对话中以关心口吻询问用户。

变更记录：
  v2 — check_conflicts() 改用 get_l1_by_id(l1_id) 直接按主键查询 L1，
       修复审查报告问题2（并发竞态风险）。
       旧写法：get_latest_l1() + id 比对
       新写法：get_l1_by_id(l1_id) 直接查，不存在则返回 None

  v3 — _call_local_model() 改用 LOCAL_MAX_TOKENS_SUMMARY（来自上轮修复）

核心流程：
  1. 读取最新 L1 摘要 + 当前生效的 L3 用户画像
  2. 调用本地 Qwen 模型做语义比对，返回冲突列表（JSON 数组）
  3. 将每条冲突写入 conflict_queue 表，状态为 pending
  4. 对外暴露两个接口供 main.py 调用：
       - get_conflict_question()  查询是否有待确认冲突，有则返回询问文本
       - handle_conflict_reply()  处理用户的"更新"/"忽略"回复

设计原则：
  - 宁可漏报，不要误报：误报会打扰用户，损害信任感
  - 不在实时对话中触发检测，只在 L1 生成后的后台流程里触发
  - 不直接修改任何记忆层，所有写操作都经过用户确认

与其他模块的关系：
  - 被调用：summarizer.generate_l1_summary() 在 L1 写入后调用 check_conflicts()
  - 被调用：main.py 的 /chat 路由调用 get_conflict_question() 和 handle_conflict_reply()
  - 读画像：database.get_current_profile()
  - 读 L1：database.get_l1_by_id()（已替换旧的 get_latest_l1）
  - 写冲突：database.save_conflict()
  - 改冲突状态：database.resolve_conflict() / database.ignore_conflict()
  - 改画像：database.update_profile_field()
  - 读配置：config.py

使用方法：
    from conflict_checker import check_conflicts, get_conflict_question, handle_conflict_reply

    # L1 生成后由 summarizer 调用
    check_conflicts(l1_id)

    # /chat 路由在构建回复前调用，有待确认冲突时返回询问文本，否则返回 None
    question = get_conflict_question()

    # 用户回复"更新"或"忽略"时，由 /chat 路由调用
    handle_conflict_reply(conflict_id, action="resolve")   # 或 action="ignore"
"""

import json
import re
import requests

from config import (
    CONFLICT_CHECK_PROMPT,
    CONFLICT_ASK_TEMPLATE,
)
from llm_client import call_local_summary
from database import (
    get_current_profile,
    get_l1_by_id,           # [修改] 替换旧的 get_latest_l1，直接按主键查询
    get_pending_conflicts,
    save_conflict,
    resolve_conflict,
    ignore_conflict,
    update_profile_field,
)


# =============================================================================
# 用户画像格式化
# =============================================================================

# 板块名到中文标签的映射，与 prompt_builder.py 保持一致
_FIELD_LABELS = {
    "basic_info":      "基础信息",
    "personal_status": "近期状态",
    "interests":       "兴趣爱好",
    "social":          "社交情况",
    "history":         "重要经历",
    "recent_context":  "近期背景",
}


def _format_profile_for_check(profile_dict):
    """
    将用户画像字典格式化为适合冲突检测 Prompt 的文本。

    与 prompt_builder._format_profile 的区别：
        这里需要明确标注板块名（英文 key），方便模型在输出冲突时
        准确填写 field 字段，而不会用中文标签导致后续匹配失败。

    参数：
        profile_dict — get_current_profile() 返回的字典

    返回：
        str — 格式化后的文本；profile_dict 为空时返回空字符串

    示例输出：
        [basic_info 基础信息] 烧酒，19岁，学生
        [personal_status 近期状态] 最近在做毕业设计，压力较大
    """
    if not profile_dict:
        return ""

    lines = []
    for field_key, label in _FIELD_LABELS.items():
        content = profile_dict.get(field_key, "").strip()
        if content:
            lines.append(f"[{field_key} {label}] {content}")

    return "\n".join(lines)


# =============================================================================
# 解析模型输出
# =============================================================================

def _strip_thinking(raw_text):
    """
    剥离模型输出中的思考链内容，只保留 JSON 部分。

    支持两种格式：
      - 标签格式：<think>...</think>
      - 纯文本格式：[ 之前的所有前缀内容（冲突检测输出是 JSON 数组，找 [）

    参数：
        raw_text — 模型返回的原始文本

    返回：
        str — 剥离思考链后的文本（已去除首尾空白）
    """
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text,
                     flags=re.DOTALL | re.IGNORECASE)

    # 冲突检测输出是 JSON 数组，找 [ 而不是 {
    bracket_pos = cleaned.find('[')
    if bracket_pos > 0:
        cleaned = cleaned[bracket_pos:]

    return cleaned.strip()


def _parse_conflict_json(raw_text):
    """
    从模型返回的原始文本中解析冲突列表。

    模型应输出 JSON 数组：有冲突时包含若干冲突对象，无冲突时输出空数组 []。

    解析策略（三步递进）：
      第一步：剥离思考链后直接解析
      第二步：正则提取第一个完整 JSON 数组

    参数：
        raw_text — 模型返回的原始文本

    返回：
        list — 冲突对象列表；无冲突时返回 []；解析失败时返回 None
    """
    # 第一步：剥离思考链
    stripped = _strip_thinking(raw_text)
    if stripped != raw_text.strip():
        print(f"[conflict_checker] 检测到思考链输出，已自动剥离")

    # 第二步：直接尝试解析
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            print(f"[conflict_checker] 警告：模型返回了对象而非数组，已自动包装")
            return [result]
    except json.JSONDecodeError:
        pass

    # 第三步：正则提取第一个完整 JSON 数组
    match = re.search(r'\[.*?\]', stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    print(f"[conflict_checker] 警告：无法解析模型输出\n原始输出：{raw_text[:200]}")
    return None


def _validate_conflict_item(item):
    """
    校验单条冲突对象的字段完整性。

    合法的板块名列表与 user_profile 表的 field 合法值保持一致。
    field 不合法的冲突条目会被丢弃，避免写入无法匹配的板块名。

    参数：
        item — _parse_conflict_json() 返回列表中的单个字典

    返回：
        bool — True 表示校验通过，False 表示此条应丢弃
    """
    valid_fields = set(_FIELD_LABELS.keys())

    for key in ("field", "old_content", "new_content", "conflict_desc"):
        if not str(item.get(key, "")).strip():
            print(f"[conflict_checker] 警告：冲突条目缺少字段 {key!r}，已丢弃")
            return False

    if item["field"] not in valid_fields:
        print(f"[conflict_checker] 警告：field 值 {item['field']!r} 不合法，已丢弃")
        return False

    return True


# =============================================================================
# 对外接口
# =============================================================================

def check_conflicts(l1_id):
    """
    对指定的 L1 摘要执行冲突检测，将发现的冲突写入 conflict_queue 表。
    这是冲突检测的入口函数，由 summarizer.generate_l1_summary() 在 L1 写入后调用。

    完整流程：
      1. 按主键读取指定 L1 的摘要内容
      2. 读取当前生效的用户画像
      3. 画像为空时跳过检测（冷启动阶段）
      4. 格式化画像和 L1 内容，填入 Prompt
      5. 调用本地模型做语义比对
      6. 解析并校验冲突列表
      7. 将每条有效冲突写入 conflict_queue 表

    参数：
        l1_id — 刚生成的 L1 记录 id（int）

    返回：
        int  — 本次检测写入的冲突条数（0 表示无冲突或检测跳过）
        None — 模型调用失败时返回 None
    """
    print(f"[conflict_checker] 开始对 L1 {l1_id} 执行冲突检测")

    # ------------------------------------------------------------------
    # 第一步：按主键读取这条 L1 的摘要内容
    #
    # [修改说明] 审查报告问题2（并发竞态风险）：
    #   旧写法：
    #     l1_row = get_latest_l1()
    #     if l1_row is None or l1_row["id"] != l1_id:
    #         ...  # 用最新一条做 id 比对校验
    #   问题：如果两次 L1 写入几乎同时触发（多 session 并发场景），
    #         get_latest_l1() 可能返回另一个更新的 L1，导致本次检测被跳过。
    #
    #   新写法：直接按主键查询，id 精确对应，不存在时返回 None，
    #   彻底消除竞态风险，逻辑也更清晰。
    # ------------------------------------------------------------------
    l1_row = get_l1_by_id(l1_id)

    if l1_row is None:
        print(f"[conflict_checker] 警告：找不到 L1 id={l1_id}，跳过冲突检测")
        return 0

    l1_summary = l1_row["summary"]
    print(f"[conflict_checker] 读取 L1 摘要：{l1_summary[:80]}")

    # 第二步：读取当前生效的用户画像
    profile_dict = get_current_profile()

    if not profile_dict:
        print(f"[conflict_checker] 用户画像为空，跳过冲突检测（冷启动阶段）")
        return 0

    # 第三步：格式化内容，填入 Prompt
    profile_text = _format_profile_for_check(profile_dict)
    prompt = CONFLICT_CHECK_PROMPT.format(
        profile_text = profile_text,
        l1_summary   = l1_summary,
    )

    # 第四步：调用本地模型
    raw_output = call_local_summary(
        messages=[{"role": "user", "content": prompt}],
        caller="conflict_checker",
    )

    if raw_output is None:
        print(f"[conflict_checker] 模型调用失败，冲突检测中止")
        return None

    print(f"[conflict_checker] 模型原始输出：{raw_output[:200]}")

    # 第五步：解析冲突列表
    conflicts = _parse_conflict_json(raw_output)

    if conflicts is None:
        print(f"[conflict_checker] 冲突列表解析失败，本次检测结果丢弃")
        return 0

    if not conflicts:
        print(f"[conflict_checker] 未检测到冲突")
        return 0

    print(f"[conflict_checker] 检测到 {len(conflicts)} 条潜在冲突，开始写入")

    # 第六步：校验并写入每条冲突
    written = 0
    for item in conflicts:
        if not _validate_conflict_item(item):
            continue

        conflict_id = save_conflict(
            source_l1_id  = l1_id,
            field         = item["field"],
            old_content   = item["old_content"].strip(),
            new_content   = item["new_content"].strip(),
            conflict_desc = item["conflict_desc"].strip(),
        )
        print(f"[conflict_checker] 冲突已写入 id={conflict_id}，field={item['field']}")
        written += 1

    print(f"[conflict_checker] 本次共写入 {written} 条冲突")
    return written


def get_conflict_question():
    """
    查询是否有待确认的冲突，有则返回第一条冲突的询问文本和 id。
    由 main.py 的 /chat 路由在构建助手回复之前调用。

    策略：
        每次只取最早的一条 pending 冲突处理，避免一次性抛出多个问题让用户困惑。

    返回：
        (int, str) — (conflict_id, 询问文本)，有待确认冲突时返回
        None       — 没有待确认冲突时返回 None
    """
    pending = get_pending_conflicts()

    if not pending:
        return None

    conflict = pending[0]
    question = CONFLICT_ASK_TEMPLATE.format(
        conflict_desc = conflict["conflict_desc"]
    )

    return conflict["id"], question


def handle_conflict_reply(conflict_id, action):
    """
    处理用户对冲突询问的回复，根据 action 决定更新记忆还是忽略。
    由 main.py 的 /chat 路由在识别到用户回复"更新"或"忽略"后调用。

    参数：
        conflict_id — conflict_queue 表的记录 id
        action      — 用户的选择：
                        "resolve" — 接受新内容，更新对应画像板块
                        "ignore"  — 保留现状，新旧两个版本均不删除

    返回：
        str — 操作结果的简短描述，供 main.py 拼入助手回复文本
    """
    pending = get_pending_conflicts()
    target = next((c for c in pending if c["id"] == conflict_id), None)

    if target is None:
        print(f"[conflict_checker] 警告：找不到 conflict_id={conflict_id}，可能已处理")
        return "这条记录好像已经处理过了。"

    field_label = _FIELD_LABELS.get(target["field"], target["field"])

    if action == "resolve":
        update_profile_field(
            field        = target["field"],
            new_content  = target["new_content"],
            source_l1_id = target["source_l1_id"],
        )
        resolve_conflict(conflict_id)
        print(f"[conflict_checker] 冲突 {conflict_id} 已 resolve，画像 {target['field']} 已更新")
        return f'好的，我已经把"{field_label}"更新成新的情况了。'

    elif action == "ignore":
        ignore_conflict(conflict_id)
        print(f"[conflict_checker] 冲突 {conflict_id} 已 ignore，画像保持不变")
        return f'好的，我会继续记住之前的"{field_label}"，这次的变化先不更新。'

    else:
        print(f"[conflict_checker] 警告：未知 action {action!r}，不做任何操作")
        return '没有理解你的选择，可以回复"更新"或"忽略"。'


# =============================================================================
# 直接运行此文件时：验证测试
# =============================================================================

if __name__ == "__main__":
    from database import (
        new_session, save_message, close_session,
        save_l1_summary, update_profile_field, get_l1_by_id as _check_get_l1,
    )

    print("=== conflict_checker.py 验证测试 ===\n")

    print("--- 测试一：画像格式化 ---")
    mock_profile = {
        "basic_info":      "烧酒，19岁，本科在读",
        "personal_status": "最近在做毕业设计，压力较大",
        "interests":       "编程、音乐",
    }
    print(_format_profile_for_check(mock_profile))
    print()

    print("--- 测试二：冲突 JSON 解析 ---")
    mock_conflict = '''[{"field":"personal_status","old_content":"压力大","new_content":"压力小","conflict_desc":"之前说压力大，现在好像轻松了？"}]'''
    r1 = _parse_conflict_json(mock_conflict)
    print(f"有冲突：{len(r1)} 条（应为 1），field={r1[0]['field']}")
    print(f"无冲突：{_parse_conflict_json('[]')}（应为 []）")
    mock_think = '<think>分析。</think>\n[{"field":"personal_status","old_content":"压力大","new_content":"压力小","conflict_desc":"变化了？"}]'
    print(f"带思考链：{len(_parse_conflict_json(mock_think))} 条（应为 1）")
    print()

    print("--- 测试三：字段校验 ---")
    valid = {"field":"personal_status","old_content":"旧","new_content":"新","conflict_desc":"变了？"}
    print(f"合法：{_validate_conflict_item(valid)}（应 True）")
    print(f"非法 field：{_validate_conflict_item({**valid,'field':'mood'})}（应 False）")
    print(f"缺字段：{_validate_conflict_item({'field':'personal_status'})}（应 False）")
    print()

    print("--- 测试四：get_l1_by_id 验证（问题2修复核心）---")
    sid_t = new_session()
    save_message(sid_t, "user", "验证按主键查询")
    close_session(sid_t)
    l1_test = save_l1_summary(sid_t, "验证摘要。", "测试", "夜间", "专注高效")
    row = _check_get_l1(l1_test)
    print(f"get_l1_by_id({l1_test}) → id={row['id']}（应为 {l1_test}）")
    print(f"get_l1_by_id(99999)   → {_check_get_l1(99999)}（应为 None）")
    print()

    print("--- 测试五：冲突队列读写 ---")
    update_profile_field("personal_status", "最近在做毕业设计，压力较大")
    sid = new_session()
    save_message(sid, "user", "今天答辩完了，轻松很多")
    close_session(sid)
    l1_id = save_l1_summary(sid, "烧酒完成毕业答辩，心情轻松愉快。",
                             "毕业,答辩,轻松", "下午", "轻松愉快")

    from database import save_conflict as _db_save
    cid = _db_save(l1_id, "personal_status", "压力较大", "心情轻松",
                   "之前说压力大，今天好像轻松很多了？")
    r = get_conflict_question()
    if r:
        print(f"get_conflict_question() → conflict_id={r[0]}")
    print(f"handle_conflict_reply(resolve) → {handle_conflict_reply(cid, 'resolve')}")
    print(f"处理后队列 → {get_conflict_question()}（应为 None）")
    print()

    print("--- 测试六：端到端（需要 LM Studio 运行中）---")
    update_profile_field("personal_status", "状态很好，精力充沛")
    sid2 = new_session()
    save_message(sid2, "user", "今天太累了，头疼")
    close_session(sid2)
    l1_id2 = save_l1_summary(sid2, "烧酒今天疲惫，头疼。", "疲惫,头疼", "夜间", "情绪低落")
    count = check_conflicts(l1_id2)
    if count is None:
        print("LM Studio 未启动，端到端跳过")
    else:
        print(f"检测到 {count} 条冲突")

    print("\n验证完成。")
