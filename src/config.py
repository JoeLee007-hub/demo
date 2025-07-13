import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 加载.env文件
load_dotenv(PROJECT_ROOT / ".env")

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
STORIES_DIR = DATA_DIR / "stories"
ANALYSIS_DIR = DATA_DIR / "analysis"
FEEDBACK_DIR = DATA_DIR / "feedback"

# 确保数据目录存在
for dir_path in [DATA_DIR, STORIES_DIR, ANALYSIS_DIR, FEEDBACK_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 模型配置
MODEL_CONFIGS = {
    "openai": {
        "default_model": "gpt-4o",
        "available_models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY"
    },
    # "claude": {
    #     "default_model": "claude-3-sonnet-20240229",
    #     "available_models": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    #     "api_key_env": "CLAUDE_API_KEY"
    # },
    # "gemini": {
    #     "default_model": "gemini-1.5-pro",
    #     "available_models": ["gemini-1.5-pro", "gemini-pro"],
    #     "api_key_env": "GEMINI_API_KEY"
    # }
}

# 应用配置
APP_CONFIG = {
    "title": "AI儿童故事创作系统",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 故事生成配置
STORY_CONFIG = {
    "min_age": 5,
    "max_age": 8,
    "default_age": 6,
    "min_word_limit": 50,
    "max_word_limit": 500,
    "default_word_limit": 300,
    "prompt_styles": ["Template", "Structured", "Question"],
    "default_character": "Little Fox",
    "default_theme": "Cooperation"
}

# 可读性分析配置
READABILITY_CONFIG = {
    "flesch_thresholds": {
        "very_easy": 90,
        "easy": 80,
        "fairly_easy": 70,
        "standard": 60,
        "fairly_difficult": 50,
        "difficult": 30,
        "very_difficult": 0
    },
    "age_recommendations": {
        "very_easy": "5-6 years",
        "easy": "6-7 years",
        "fairly_easy": "7-8 years",
        "standard": "8-9 years",
        "fairly_difficult": "9-10 years",
        "difficult": "10+ years",
        "very_difficult": "Adult"
    }
}

# 反馈配置
FEEDBACK_CONFIG = {
    "rating_range": (1, 5),
    "default_rating": 5,
    "max_comment_length": 1000
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "app.log"
}

# 确保日志目录存在
LOGGING_CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)

# 缓存配置
CACHE_CONFIG = {
    "ttl": 3600,  # 1小时
    "max_entries": 100
}

# 安全配置
SECURITY_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_file_types": [".txt", ".json", ".csv"],
    "max_story_length": 2000,  # 最大故事长度
    "max_prompt_length": 1000  # 最大prompt长度
}

# 多语言配置
LANGUAGE_CONFIG = {
    "default_language": "en",
    "supported_languages": ["zh", "en"],
    "language_names": {
        "zh": "中文 🇨🇳",
        "en": "English 🇬🇧"
    }
}

# 数据库配置（如果需要）
DATABASE_CONFIG = {
    "type": "json",  # 目前使用JSON文件存储
    "backup_enabled": True,
    "backup_interval": 24 * 3600,  # 24小时
    "max_backups": 7  # 保留7个备份
}

# 性能配置
PERFORMANCE_CONFIG = {
    "max_concurrent_requests": 5,
    "request_timeout": 30,  # 秒
    "retry_attempts": 3,
    "retry_delay": 1  # 秒
}

# 开发配置
DEV_CONFIG = {
    "debug_mode": os.getenv("DEBUG", "False").lower() == "true",
    "hot_reload": True,
    "show_error_details": True
}

# 导出所有配置
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR", "STORIES_DIR", "ANALYSIS_DIR", "FEEDBACK_DIR",
    "OPENAI_API_KEY", # "CLAUDE_API_KEY", "GEMINI_API_KEY",
    "MODEL_CONFIGS", "APP_CONFIG", "STORY_CONFIG",
    "READABILITY_CONFIG", "FEEDBACK_CONFIG", "LOGGING_CONFIG",
    "CACHE_CONFIG", "SECURITY_CONFIG", "LANGUAGE_CONFIG",
    "DATABASE_CONFIG", "PERFORMANCE_CONFIG", "DEV_CONFIG"
]