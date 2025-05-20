"""
Story Generator Module
AI增强儿童故事创作系统 - 故事生成模块

功能：
- 支持多模型（如openai、claude、gemini）灵活切换
- 各模型API Key均从环境变量读取，安全管理
- 调用OpenAI API或其他模型API，根据Prompt生成故事文本
- 支持模型选择、参数自定义
- 便于后续集成到主系统
"""

import os
# Streamlit Cloud secrets 兼容
try:
    import streamlit as st
    secrets = st.secrets
except ImportError:
    secrets = {}

import openai

class StoryGenerator:
    """
    故事生成主类
    支持通过Prompt调用多种模型生成故事
    """
    def __init__(self, model: str = "openai", openai_model: str = "gpt-3.5-turbo"):
        self.model = model
        self.openai_model = openai_model
        # 优先用secrets，其次用环境变量
        self.api_keys = {
            "openai": secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            "claude": secrets.get("CLAUDE_API_KEY", os.getenv("CLAUDE_API_KEY")),
            "gemini": secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
        }
        # 初始化各模型API
        if self.model == "openai":
            self.openai_client = openai.OpenAI(api_key=self.api_keys["openai"])
        # 其他模型初始化可在此扩展

    def generate_story(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        根据Prompt生成故事文本
        :param prompt: 输入的Prompt字符串
        :param temperature: 创意性参数
        :param max_tokens: 最大生成长度
        :return: 生成的故事文本
        """
        if self.model == "openai":
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                story = response.choices[0].message.content.strip()
                return story
            except Exception as e:
                print(f"[错误] OpenAI故事生成失败: {e}")
                return "[生成失败，请检查OpenAI API Key和网络设置]"
        elif self.model == "claude":
            # TODO: 集成Claude API调用
            return "[Claude模型暂未集成]"
        elif self.model == "gemini":
            # TODO: 集成Gemini API调用
            return "[Gemini模型暂未集成]"
        else:
            return "[不支持的模型类型]"

# 示例用法
if __name__ == "__main__":
    generator = StoryGenerator(model="openai", openai_model="gpt-3.5-turbo")
    sample_prompt = "请为6岁的儿童写一个关于\"勇敢\"的原创故事，主角是一只小兔子，语言简洁，积极正面，字数不超过300词。"
    print("生成故事示例：\n", generator.generate_story(sample_prompt)) 