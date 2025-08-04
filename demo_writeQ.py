import json

with open('./data/7215_gold.json', 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

questions = [{"question": item["question"]} for item in gold_data]

with open('./data/7215_question.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)
