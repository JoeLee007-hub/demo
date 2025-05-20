"""
Readability Analysis Module
AI增强儿童故事创作系统 - 可读性分析模块

功能：
- 基于textstat库，输出Flesch Reading Ease、平均句长、平均词长等指标
- 提供适合年龄段建议
- 便于后续集成到主系统
"""

import textstat
from typing import Dict

class ReadabilityAnalyzer:
    """
    可读性分析主类
    支持多项可读性指标分析
    """
    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict:
        """
        分析文本可读性
        :param text: 输入文本
        :return: 可读性分析结果字典
        """
        fre = textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        avg_syllables_per_word = textstat.avg_syllables_per_word(text)
        word_count = textstat.lexicon_count(text, removepunct=True)
        sentence_count = textstat.sentence_count(text)
        if fre >= 90:
            age_range = "5-6"
        elif fre >= 80:
            age_range = "6-8"
        elif fre >= 70:
            age_range = "8-10"
        else:
            age_range = ">10"
        return {
            "Flesch Reading Ease": fre,
            "Average Sentence Length": avg_sentence_length,
            "Average Syllables per Word": avg_syllables_per_word,
            "Word Count": word_count,
            "Sentence Count": sentence_count,
            "Recommended Age Range": age_range
        }

# 示例用法
if __name__ == "__main__":
    analyzer = ReadabilityAnalyzer()
    sample_text = "小兔子很勇敢。一天，它帮助了小鸟。大家都很开心。"
    result = analyzer.analyze(sample_text)
    print("可读性分析结果：")
    for k, v in result.items():
        print(f"{k}: {v}") 