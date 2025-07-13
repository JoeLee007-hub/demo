"""
Feedback Collection & Archiving Module
AI增强儿童故事创作系统 - 用户反馈与数据归档模块

功能：
- 采集用户对故事的评分与主观评价
- 反馈数据本地保存为JSON文件，便于后续分析
- 便于后续集成到主系统
"""

import json
import os
from typing import Dict, Any
from datetime import datetime

class FeedbackCollector:
    """
    用户反馈采集与归档主类
    支持评分、主观评价采集及本地JSON归档
    """
    def __init__(self, save_dir: str = "data/feedback"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def collect_feedback(self, story_id: str, rating: int, comment: str, extra: Dict[str, Any] = None) -> str:
        """
        采集并保存用户反馈
        :param story_id: 故事唯一标识
        :param rating: 用户评分（如1-5分）
        :param comment: 主观评价
        :param extra: 其他附加信息（可选）
        :return: 保存的文件路径
        """
        feedback = {
            "story_id": story_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            feedback.update(extra)
        filename = f"{story_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)
        return filepath

# 示例用法
if __name__ == "__main__":
    collector = FeedbackCollector()
    path = collector.collect_feedback(
        story_id="story001",
        rating=5,
        comment="故事很有教育意义，孩子很喜欢！",
        extra={"user_type": "parent"}
    )
    print(f"反馈已保存至: {path}") 