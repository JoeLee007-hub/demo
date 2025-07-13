import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

# 设置日志
logger = logging.getLogger(__name__)

def generate_uuid() -> str:
    """生成唯一标识符"""
    return str(uuid.uuid4())

def get_current_timestamp() -> str:
    """获取当前时间戳（ISO格式）"""
    return datetime.now().isoformat()

def safe_json_load(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """安全地加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"JSON文件不存在: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {file_path}, 错误: {e}")
        return None
    except Exception as e:
        logger.error(f"读取JSON文件时发生未知错误: {file_path}, 错误: {e}")
        return None

def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """安全地保存JSON文件"""
    try:
        # 确保目录存在
        ensure_directory(Path(file_path).parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存JSON文件时发生错误: {file_path}, 错误: {e}")
        return False

def ensure_directory(dir_path: Union[str, Path]) -> bool:
    """确保目录存在，如果不存在则创建"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录时发生错误: {dir_path}, 错误: {e}")
        return False

def get_file_size(file_path: Union[str, Path]) -> int:
    """获取文件大小（字节）"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def count_words(text: str) -> int:
    """计算文本中的单词数"""
    if not text or not text.strip():
        return 0
    return len(text.split())

def count_sentences(text: str) -> int:
    """计算文本中的句子数"""
    if not text or not text.strip():
        return 0
    
    # 简单的句子分割（基于句号、问号、感叹号）
    import re
    sentences = re.split(r'[.!?]+', text)
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def calculate_average_word_length(text: str) -> float:
    """计算平均单词长度"""
    if not text or not text.strip():
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    total_length = sum(len(word.strip('.,!?;:"()[]{}')) for word in words)
    return total_length / len(words)

def calculate_average_sentence_length(text: str) -> float:
    """计算平均句子长度（单词数）"""
    word_count = count_words(text)
    sentence_count = count_sentences(text)
    
    if sentence_count == 0:
        return 0.0
    
    return word_count / sentence_count

def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """从文本中提取关键词（简单实现）"""
    if not text or not text.strip():
        return []
    
    # 简单的关键词提取：去除常见停用词，统计词频
    import re
    from collections import Counter
    
    # 基本的英文停用词
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
        'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # 清理文本并分词
    words = re.findall(r'\b\w+\b', text.lower())
    
    # 过滤停用词和短词
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # 统计词频并返回最常见的词
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(max_keywords)]

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个字典"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """扁平化嵌套字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    import time
    
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"函数 {func.__name__} 在 {max_retries} 次尝试后仍然失败")
        
        raise last_exception
    
    return wrapper

def batch_process(items: list, batch_size: int = 10):
    """批量处理列表项"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def is_valid_url(url: str) -> bool:
    """检查URL是否有效"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def get_memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil未安装，无法获取内存使用情况")
        return {}
    except Exception as e:
        logger.error(f"获取内存使用情况时发生错误: {e}")
        return {}