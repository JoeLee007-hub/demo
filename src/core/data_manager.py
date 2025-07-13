import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class DataManager:
    def __init__(self):
        # 确保数据目录存在
        self.data_dir = "data"
        self.stories_dir = os.path.join(self.data_dir, "stories")
        self.feedback_dir = os.path.join(self.data_dir, "user_feedback")
        self._init_directories()

    def _init_directories(self):
        """初始化必要的数据目录"""
        for dir_path in [self.stories_dir, self.feedback_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def save_story(self, story_data: Dict, story_id: Optional[str] = None) -> str:
        """保存生成的故事"""
        if not story_id:
            story_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = os.path.join(self.stories_dir, f"{story_id}.json")
        
        # 添加元数据
        story_data.update({
            "created_at": datetime.now().isoformat(),
            "story_id": story_id
        })
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, ensure_ascii=False, indent=2)
        
        return story_id

    def get_story(self, story_id: str) -> Optional[Dict]:
        """获取指定故事"""
        file_path = os.path.join(self.stories_dir, f"{story_id}.json")
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_stories(self) -> List[Dict]:
        """列出所有故事"""
        stories = []
        for file_name in os.listdir(self.stories_dir):
            if file_name.endswith(".json"):
                story_id = file_name[:-5]  # 移除.json后缀
                story = self.get_story(story_id)
                if story:
                    stories.append(story)
        return stories

    def save_feedback(self, feedback_data: Dict, feedback_id: Optional[str] = None) -> str:
        """保存用户反馈"""
        if not feedback_id:
            feedback_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = os.path.join(self.feedback_dir, f"{feedback_id}.json")
        
        feedback_data.update({
            "created_at": datetime.now().isoformat(),
            "feedback_id": feedback_id
        })
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        return feedback_id

    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        """获取指定反馈"""
        file_path = os.path.join(self.feedback_dir, f"{feedback_id}.json")
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_feedback(self) -> List[Dict]:
        """列出所有反馈"""
        feedback_list = []
        for file_name in os.listdir(self.feedback_dir):
            if file_name.endswith(".json"):
                feedback_id = file_name[:-5]  # 移除.json后缀
                feedback = self.get_feedback(feedback_id)
                if feedback:
                    feedback_list.append(feedback)
        return feedback_list

# 使用示例：
"""
# 初始化数据管理器
data_manager = DataManager()

# 保存故事
story_data = {
    "title": "小红帽的冒险",
    "content": "从前有一个小女孩...",
    "language": "zh",
    "tags": ["童话", "冒险"]
}
story_id = data_manager.save_story(story_data)

# 获取故事
story = data_manager.get_story(story_id)

# 保存反馈
feedback_data = {
    "story_id": story_id,
    "rating": 5,
    "comment": "很有趣的故事！"
}
data_manager.save_feedback(feedback_data)
"""