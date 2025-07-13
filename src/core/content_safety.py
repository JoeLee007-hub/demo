# -*- coding: utf-8 -*-
"""
内容安全检查模块
提供儿童内容创作系统的多层次安全检查功能
"""

import re
from typing import Dict, List, Tuple, Optional
from src.utils import sanitize_text_input

class ContentSafetyChecker:
    """
    内容安全检查器
    实现多层次的内容安全检查机制
    """
    
    def __init__(self):
        # 不适宜词汇列表（示例，实际应用中需要更完整的列表）
        self.inappropriate_words = {
            'zh': ['暴力', '恐怖', '血腥', '死亡', '杀害', '伤害', '欺凌', '歧视'],
            'en': ['violence', 'terror', 'blood', 'death', 'kill', 'harm', 'bully', 'discriminate']
        }
        
        # 积极词汇列表
        self.positive_words = {
            'zh': ['友谊', '合作', '善良', '勇敢', '诚实', '分享', '帮助', '关爱'],
            'en': ['friendship', 'cooperation', 'kindness', 'brave', 'honest', 'share', 'help', 'care']
        }
    
    def check_input_safety(self, text: str, lang: str = 'zh') -> Dict:
        """
        检查用户输入的安全性
        
        Args:
            text: 待检查的文本
            lang: 语言代码 ('zh' 或 'en')
            
        Returns:
            Dict: 包含安全检查结果的字典
        """
        result = {
            'is_safe': True,
            'issues': [],
            'cleaned_text': text,
            'warnings': []
        }
        
        if not text or not text.strip():
            return result
        
        # 基础文本清理
        cleaned_text = sanitize_text_input(text, max_length=200)
        
        # 检查是否包含不适宜词汇
        inappropriate_found = self._check_inappropriate_words(cleaned_text, lang)
        if inappropriate_found:
            result['is_safe'] = False
            result['issues'].extend(inappropriate_found)
            result['warnings'].append('检测到不适宜内容' if lang == 'zh' else 'Inappropriate content detected')
        
        # 检查文本长度
        if len(cleaned_text) > 200:
            result['warnings'].append('文本已截断至安全长度' if lang == 'zh' else 'Text truncated to safe length')
        
        result['cleaned_text'] = cleaned_text
        return result
    
    def check_story_safety(self, story: str, lang: str = 'zh') -> Dict:
        """
        检查生成故事的安全性
        
        Args:
            story: 待检查的故事文本
            lang: 语言代码
            
        Returns:
            Dict: 包含安全检查结果的字典
        """
        result = {
            'is_safe': True,
            'safety_score': 0.0,
            'issues': [],
            'cleaned_story': story,
            'recommendations': []
        }
        
        if not story or not story.strip():
            return result
        
        # 基础文本清理
        cleaned_story = sanitize_text_input(story, max_length=2000)
        
        # 检查不适宜内容
        inappropriate_found = self._check_inappropriate_words(cleaned_story, lang)
        if inappropriate_found:
            result['is_safe'] = False
            result['issues'].extend(inappropriate_found)
        
        # 计算安全评分
        result['safety_score'] = self._calculate_safety_score(cleaned_story, lang)
        
        # 生成建议
        if result['safety_score'] < 0.7:
            result['recommendations'].append(
                '建议增加更多积极正面的内容' if lang == 'zh' else 'Recommend adding more positive content'
            )
        
        result['cleaned_story'] = cleaned_story
        return result
    
    def _check_inappropriate_words(self, text: str, lang: str) -> List[str]:
        """
        检查文本中的不适宜词汇
        
        Args:
            text: 待检查的文本
            lang: 语言代码
            
        Returns:
            List[str]: 发现的不适宜词汇列表
        """
        found_words = []
        inappropriate_list = self.inappropriate_words.get(lang, [])
        
        text_lower = text.lower()
        for word in inappropriate_list:
            if word.lower() in text_lower:
                found_words.append(word)
        
        return found_words
    
    def _calculate_safety_score(self, text: str, lang: str) -> float:
        """
        计算文本的安全评分 (0-1，1为最安全)
        
        Args:
            text: 待评分的文本
            lang: 语言代码
            
        Returns:
            float: 安全评分
        """
        if not text:
            return 1.0
        
        score = 1.0
        text_lower = text.lower()
        
        # 检查不适宜词汇（每个扣0.2分）
        inappropriate_list = self.inappropriate_words.get(lang, [])
        for word in inappropriate_list:
            if word.lower() in text_lower:
                score -= 0.2
        
        # 检查积极词汇（每个加0.1分，最多加0.3分）
        positive_list = self.positive_words.get(lang, [])
        positive_count = 0
        for word in positive_list:
            if word.lower() in text_lower:
                positive_count += 1
        
        score += min(positive_count * 0.1, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def get_content_recommendations(self, text: str, lang: str = 'zh') -> List[str]:
        """
        获取内容改进建议
        
        Args:
            text: 文本内容
            lang: 语言代码
            
        Returns:
            List[str]: 改进建议列表
        """
        recommendations = []
        
        if not text or len(text.strip()) < 50:
            recommendations.append(
                '建议增加更多故事内容' if lang == 'zh' else 'Recommend adding more story content'
            )
        
        # 检查是否包含教育元素
        educational_keywords = {
            'zh': ['学习', '成长', '友谊', '分享', '帮助'],
            'en': ['learn', 'grow', 'friendship', 'share', 'help']
        }
        
        has_educational = False
        for keyword in educational_keywords.get(lang, []):
            if keyword.lower() in text.lower():
                has_educational = True
                break
        
        if not has_educational:
            recommendations.append(
                '建议添加更多教育性内容' if lang == 'zh' else 'Recommend adding more educational content'
            )
        
        return recommendations