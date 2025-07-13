import re
from datetime import datetime, timedelta
from typing import Any, Optional, Union
import logging

# 设置日志
logger = logging.getLogger(__name__)

def format_timestamp(timestamp: Union[str, datetime], format_type: str = 'readable') -> str:
    """格式化时间戳
    
    Args:
        timestamp: 时间戳（字符串或datetime对象）
        format_type: 格式类型 ('readable', 'short', 'iso', 'filename')
    """
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            return "Invalid timestamp"
        
        if format_type == 'readable':
            return dt.strftime('%Y年%m月%d日 %H:%M:%S')
        elif format_type == 'short':
            return dt.strftime('%m/%d %H:%M')
        elif format_type == 'iso':
            return dt.isoformat()
        elif format_type == 'filename':
            return dt.strftime('%Y%m%d_%H%M%S')
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
            
    except Exception as e:
        logger.error(f"格式化时间戳时发生错误: {e}")
        return "Invalid timestamp"

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def format_duration(seconds: Union[int, float]) -> str:
    """格式化持续时间"""
    try:
        seconds = int(seconds)
        
        if seconds < 60:
            return f"{seconds}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            if remaining_seconds == 0:
                return f"{minutes}分钟"
            else:
                return f"{minutes}分{remaining_seconds}秒"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            if remaining_minutes == 0:
                return f"{hours}小时"
            else:
                return f"{hours}小时{remaining_minutes}分钟"
                
    except Exception as e:
        logger.error(f"格式化持续时间时发生错误: {e}")
        return "未知时长"

def format_rating_stars(rating: Union[int, float]) -> str:
    """将评分转换为星星显示"""
    try:
        rating = float(rating)
        full_stars = int(rating)
        half_star = 1 if rating - full_stars >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        
        return "★" * full_stars + "★" * half_star + "☆" * empty_stars
    except Exception:
        return "☆☆☆☆☆"

def format_readability_score(score: Union[int, float]) -> str:
    """格式化可读性分数并添加描述"""
    try:
        score = float(score)
        
        if score >= 90:
            level = "非常容易"
            color = ""
        elif score >= 80:
            level = "容易"
            color = ""
        elif score >= 70:
            level = "较容易"
            color = ""
        elif score >= 60:
            level = "标准"
            color = ""
        elif score >= 50:
            level = "较困难"
            color = ""
        elif score >= 30:
            level = "困难"
            color = ""
        else:
            level = "非常困难"
            color = ""
        
        return f"{color} {score:.1f} ({level})"
        
    except Exception:
        return "❓ 未知"

def format_percentage(value: Union[int, float], total: Union[int, float]) -> str:
    """格式化百分比"""
    try:
        if total == 0:
            return "0.0%"
        
        percentage = (value / total) * 100
        return f"{percentage:.1f}%"
        
    except Exception:
        return "0.0%"

def format_number_with_commas(number: Union[int, float]) -> str:
    """为数字添加千位分隔符"""
    try:
        if isinstance(number, float):
            return f"{number:,.2f}"
        else:
            return f"{number:,}"
    except Exception:
        return str(number)

def format_age_range(min_age: int, max_age: int) -> str:
    """格式化年龄范围"""
    if min_age == max_age:
        return f"{min_age}岁"
    else:
        return f"{min_age}-{max_age}岁"

def format_word_count(count: int) -> str:
    """格式化词数显示"""
    if count < 1000:
        return f"{count}词"
    elif count < 1000000:
        return f"{count/1000:.1f}K词"
    else:
        return f"{count/1000000:.1f}M词"

def clean_text(text: str) -> str:
    """清理文本，移除多余的空白字符"""
    if not isinstance(text, str):
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除开头和结尾的空白
    text = text.strip()
    
    return text

def format_story_preview(content: str, max_length: int = 100) -> str:
    """格式化故事预览"""
    if not content:
        return "暂无内容"
    
    # 清理文本
    content = clean_text(content)
    
    # 截断文本
    if len(content) <= max_length:
        return content
    
    # 尝试在句子边界截断
    truncated = content[:max_length]
    last_sentence_end = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?'),
        truncated.rfind('。'),
        truncated.rfind('！'),
        truncated.rfind('？')
    )
    
    if last_sentence_end > max_length * 0.7:  # 如果句子边界不太远
        return truncated[:last_sentence_end + 1]
    else:
        return truncated + "..."

def format_model_name(model: str) -> str:
    """格式化模型名称显示"""
    model_display_names = {
        'gpt-4o': 'GPT-4o',
        # 'claude-3': 'Claude-3',
        # 'gemini-1.5-pro': 'Gemini-1.5-Pro',
        'gpt-4': 'GPT-4',
        'gpt-3.5-turbo': 'GPT-3.5-Turbo'
    }
    
    return model_display_names.get(model, model.upper())

def format_prompt_style(style: str) -> str:
    """格式化Prompt风格显示"""
    style_display_names = {
        'Template': '模板式',
        'Structured': '结构式', 
        'Question': '问句式',
        '模板式': 'Template',
        '结构式': 'Structured',
        '问句式': 'Question'
    }
    
    return style_display_names.get(style, style)

def format_theme_display(theme: str) -> str:
    """格式化主题显示"""
    if not theme:
        return "未指定主题"
    
    # 首字母大写
    return theme.strip().title()

def format_character_display(character: str) -> str:
    """格式化角色显示"""
    if not character:
        return "未指定角色"
    
    return character.strip()

def format_language_display(lang_code: str) -> str:
    """格式化语言显示"""
    language_names = {
        'zh': '中文',
        'en': 'English',
        'zh-CN': '简体中文',
        'zh-TW': '繁体中文',
        'en-US': 'English (US)',
        'en-GB': 'English (UK)'
    }
    
    return language_names.get(lang_code, lang_code.upper())

def format_json_pretty(data: Any, indent: int = 2) -> str:
    """格式化JSON数据为美观的字符串"""
    import json
    
    try:
        return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True)
    except Exception as e:
        logger.error(f"格式化JSON时发生错误: {e}")
        return str(data)

def format_error_message(error: Exception, context: str = "") -> str:
    """格式化错误消息"""
    error_type = type(error).__name__
    error_message = str(error)
    
    if context:
        return f"[{context}] {error_type}: {error_message}"
    else:
        return f"{error_type}: {error_message}"

def format_success_message(message: str, details: str = "") -> str:
    """格式化成功消息"""
    if details:
        return f"✅ {message} - {details}"
    else:
        return f"✅ {message}"

def format_warning_message(message: str, details: str = "") -> str:
    """格式化警告消息"""
    if details:
        return f"⚠️ {message} - {details}"
    else:
        return f"⚠️ {message}"

def format_info_message(message: str, details: str = "") -> str:
    """格式化信息消息"""
    if details:
        return f"ℹ️ {message} - {details}"
    else:
        return f"ℹ️ {message}"

def format_table_data(data: list, headers: list) -> str:
    """格式化表格数据为文本"""
    if not data or not headers:
        return "无数据"
    
    # 计算每列的最大宽度
    col_widths = [len(header) for header in headers]
    
    for row in data:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # 构建表格
    lines = []
    
    # 表头
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    lines.append(header_line)
    
    # 分隔线
    separator = " | ".join("-" * width for width in col_widths)
    lines.append(separator)
    
    # 数据行
    for row in data:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)
    
    return "\n".join(lines)