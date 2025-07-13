import re
import os
from pathlib import Path
from typing import Any, List, Optional, Union
import logging

# 设置日志
logger = logging.getLogger(__name__)

def validate_age(age: Any) -> bool:
    """验证年龄是否在有效范围内（5-8岁）"""
    try:
        age_int = int(age)
        return 5 <= age_int <= 8
    except (ValueError, TypeError):
        return False

def validate_word_limit(word_limit: Any) -> bool:
    """验证字数限制是否在有效范围内（50-500）"""
    try:
        limit_int = int(word_limit)
        return 50 <= limit_int <= 500
    except (ValueError, TypeError):
        return False

def validate_rating(rating: Any) -> bool:
    """验证评分是否在有效范围内（1-5）"""
    try:
        rating_int = int(rating)
        return 1 <= rating_int <= 5
    except (ValueError, TypeError):
        return False

def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """验证文件类型是否被允许"""
    if allowed_types is None:
        allowed_types = ['.txt', '.json', '.csv', '.md']
    
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_types

def validate_file_size(file_path: Union[str, Path], max_size_mb: float = 10.0) -> bool:
    """验证文件大小是否在限制范围内"""
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    except OSError:
        return False

def validate_story_content(content: str, min_length: int = 10, max_length: int = 2000) -> bool:
    """验证故事内容是否符合要求"""
    if not isinstance(content, str):
        return False
    
    content = content.strip()
    if not content:
        return False
    
    return min_length <= len(content) <= max_length

def validate_prompt_content(prompt: str, max_length: int = 1000) -> bool:
    """验证Prompt内容是否符合要求"""
    if not isinstance(prompt, str):
        return False
    
    prompt = prompt.strip()
    if not prompt:
        return False
    
    return len(prompt) <= max_length

def validate_character_name(character: str) -> bool:
    """验证角色名称是否有效"""
    if not isinstance(character, str):
        return False
    
    character = character.strip()
    if not character:
        return False
    
    # 检查长度（1-50个字符）
    if not (1 <= len(character) <= 50):
        return False
    
    # 检查是否包含有效字符（字母、数字、空格、基本标点）
    pattern = r'^[a-zA-Z0-9\u4e00-\u9fff\s\-\'".,!?()]+$'
    return bool(re.match(pattern, character))

def validate_theme(theme: str) -> bool:
    """验证主题是否有效"""
    if not isinstance(theme, str):
        return False
    
    theme = theme.strip()
    if not theme:
        return False
    
    # 检查长度（1-100个字符）
    if not (1 <= len(theme) <= 100):
        return False
    
    # 检查是否包含有效字符
    pattern = r'^[a-zA-Z0-9\u4e00-\u9fff\s\-\'".,!?()]+$'
    return bool(re.match(pattern, theme))

def validate_comment(comment: str, max_length: int = 1000) -> bool:
    """验证评论内容是否有效"""
    if not isinstance(comment, str):
        return False
    
    # 评论可以为空
    if not comment.strip():
        return True
    
    return len(comment.strip()) <= max_length

def validate_language_code(lang_code: str) -> bool:
    """验证语言代码是否有效"""
    valid_codes = ['zh', 'en']
    return lang_code in valid_codes

def validate_model_name(model: str) -> bool:
    """验证模型名称是否有效"""
    valid_models = ['gpt-4o']  # 'claude-3', 'gemini-1.5-pro']
    return model in valid_models

def validate_prompt_style(style: str) -> bool:
    """验证Prompt风格是否有效"""
    valid_styles = ['Template', 'Structured', 'Question', '模板式', '结构式', '问句式']
    return style in valid_styles

def validate_email(email: str) -> bool:
    """验证邮箱地址格式"""
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_uuid(uuid_string: str) -> bool:
    """验证UUID格式"""
    if not isinstance(uuid_string, str):
        return False
    
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(pattern, uuid_string.lower()))

def validate_timestamp(timestamp: str) -> bool:
    """验证时间戳格式（ISO格式）"""
    if not isinstance(timestamp, str):
        return False
    
    try:
        from datetime import datetime
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False

def validate_json_structure(data: Any, required_fields: List[str]) -> bool:
    """验证JSON数据结构是否包含必需字段"""
    if not isinstance(data, dict):
        return False
    
    return all(field in data for field in required_fields)

def validate_story_data(story_data: dict) -> bool:
    """验证故事数据结构"""
    required_fields = ['story_id', 'content', 'timestamp']
    
    if not validate_json_structure(story_data, required_fields):
        return False
    
    # 验证具体字段
    if not validate_uuid(story_data['story_id']):
        return False
    
    if not validate_story_content(story_data['content']):
        return False
    
    if not validate_timestamp(story_data['timestamp']):
        return False
    
    # 验证参数（如果存在）
    if 'parameters' in story_data:
        params = story_data['parameters']
        if isinstance(params, dict):
            if 'age' in params and not validate_age(params['age']):
                return False
            if 'word_limit' in params and not validate_word_limit(params['word_limit']):
                return False
            if 'character' in params and not validate_character_name(params['character']):
                return False
            if 'theme' in params and not validate_theme(params['theme']):
                return False
    
    return True

def validate_feedback_data(feedback_data: dict) -> bool:
    """验证反馈数据结构"""
    required_fields = ['story_id', 'rating', 'timestamp']
    
    if not validate_json_structure(feedback_data, required_fields):
        return False
    
    # 验证具体字段
    if not validate_uuid(feedback_data['story_id']):
        return False
    
    if not validate_rating(feedback_data['rating']):
        return False
    
    if not validate_timestamp(feedback_data['timestamp']):
        return False
    
    # 验证评论（如果存在）
    if 'comment' in feedback_data:
        if not validate_comment(feedback_data['comment']):
            return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全字符"""
    if not isinstance(filename, str):
        return "untitled"
    
    # 移除路径分隔符和其他不安全字符
    unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(unsafe_chars, '_', filename)
    
    # 移除开头和结尾的点和空格
    sanitized = sanitized.strip('. ')
    
    # 确保文件名不为空
    if not sanitized:
        sanitized = "untitled"
    
    # 限制长度
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized

def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """清理文本输入，移除潜在的危险内容"""
    if not isinstance(text, str):
        return ""
    
    # 移除控制字符（除了常见的空白字符）
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 限制长度
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()

def validate_api_key(api_key: str) -> bool:
    """验证API密钥格式（基本检查）"""
    if not isinstance(api_key, str):
        return False
    
    api_key = api_key.strip()
    
    # 基本长度检查
    if len(api_key) < 10:
        return False
    
    # 检查是否包含明显的占位符
    placeholders = ['your_api_key', 'api_key_here', 'replace_me', 'xxx']
    if api_key.lower() in placeholders:
        return False
    
    return True

def validate_config_dict(config: dict, required_keys: List[str]) -> bool:
    """验证配置字典是否包含必需的键"""
    if not isinstance(config, dict):
        return False
    
    return all(key in config for key in required_keys)

def is_safe_path(path: Union[str, Path], base_path: Union[str, Path]) -> bool:
    """检查路径是否安全（防止路径遍历攻击）"""
    try:
        base_path = Path(base_path).resolve()
        target_path = Path(path).resolve()
        
        # 检查目标路径是否在基础路径内
        return str(target_path).startswith(str(base_path))
    except Exception:
        return False