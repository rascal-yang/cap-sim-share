"""Deterministic English words and Chinese bigrams for SQLite FTS5."""
import json
import re


def grams(text):
    words = set(re.findall(r'[a-z0-9_]+', text.lower()))
    for run in re.findall(r'[\u3400-\u9fff]+', text):
        words.update(run[i:i+2] for i in range(max(1, len(run)-1)))
    return words


def searchable(record):
    text = json.dumps({k: record[k] for k in ('task_id', 'context', 'orchestration', 'control', 'metrics')}, ensure_ascii=False)
    aliases = {'Apple': '苹果 抓取', 'Plate': '盘子 抓取', 'Button': '按钮 按压',
               'ChairSit': '椅子 坐下', 'ChairPush': '椅子 推动', 'Dolly': '推车 推动',
               'Basket': '篮子 抓取', 'Bottle': '瓶子 抓取', 'Box': '箱子 抓取', 'BallKick': '踢球',
               'contact_entry_failed': '接触失败', 'collision': '碰撞', 'navigation': '导航 站位 停稳',
               'lift': '抬升', 'opposite': '反向响应'}
    return text+' '+' '.join(v for k, v in aliases.items() if k.lower() in text.lower())
