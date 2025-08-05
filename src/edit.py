import json
import ast
import re
from pathlib import Path

# 리스트형 문자열 안전 파싱
def safe_eval_or_str(val):
    try:
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return ", ".join(map(str, parsed))
        return str(val).replace('"', '').strip()
    except:
        return str(val).replace('"', '').strip()

# HTML 태그 제거
def clean_html(text):
    return re.sub(r"<[^>]+>", "", text)

# '정보없음' 등 무의미한 값 필터링
def is_meaningful(val: str) -> bool:
    val = str(val).strip().lower()
    return val not in ["정보없음", "없음", "nan", "", "null"]

def generate_natural_metadata(data):
    summary = []

    name = data.get('캠핑장이름', '이름없음').strip('"')
    addr = data.get('캠핑장주소', '주소없음').strip('"')
    line = f"{name}은(는) {addr}에 위치한 캠핑장으로,"

    status = data.get("운영상태", "운영정보없음")
    time = data.get("운영시간", "")
    if "|" in time:
        checkin, checkout = time.split("|")
        line += f" 현재 '{status}' 중이며 입실은 오후 {checkin}, 퇴실은 오전 {checkout}입니다."
    else:
        line += f" 현재 '{status}' 중입니다."
    summary.append(line)

    description = ""
    for field in ["캠핑장설명", "캠핑장메모"]:
        val = data.get(field)
        if val and is_meaningful(val):
            description += clean_html(safe_eval_or_str(val)) + " "
    description = description.strip()
    if description:
        summary.append(f"이 캠핑장에 대한 설명으로는 '{description}'라고 안내되어 있습니다.")
    else:
        summary.append("캠핑장 설명은 별도로 제공되지 않았습니다.")

    price = data.get("가격", "")
    if "^" in price and is_meaningful(price):
        price_parts = [p.replace("|", "은 ") + "원" for p in price.split("^")]
        summary.append("요금은 " + ", ".join(price_parts) + " 정도로 확인됩니다.")
    else:
        summary.append("요금 정보는 따로 안내되어 있지 않습니다.")

    for i in range(1, 4):
        title = data.get(f"추가제목{i}", "").strip('"')
        content = data.get(f"추가내용{i}", "").replace("|", " ").strip('"')
        if is_meaningful(title) and is_meaningful(content):
            summary.append(f"{title}에 대해서는 '{clean_html(content)}'라고 소개되어 있습니다.")
        elif is_meaningful(title):
            summary.append(f"{title}에 대한 자세한 정보는 제공되지 않았습니다.")

    for label, key in [("주변환경", "주변환경"), ("지형 특성", "지형특성")]:
        val = data.get(key)
        if val and is_meaningful(val):
            parsed = safe_eval_or_str(val)
            summary.append(f"{label}으로는 {parsed} 등이 있습니다.")
        else:
            summary.append(f"{label} 관련 정보는 제공되지 않았습니다.")

    for label, key in [("이용 가능한 시설", "캠핑장시설"), ("즐길거리", "즐길거리")]:
        val = data.get(key)
        if val and is_meaningful(val):
            parsed = safe_eval_or_str(val)
            summary.append(f"{label}로는 {parsed} 등이 마련되어 있습니다.")
        else:
            summary.append(f"{label} 정보는 별도로 확인되지 않았습니다.")

    comments = data.get("캠핑장댓글", "")
    try:
        if isinstance(comments, str) and comments.startswith("["):
            comments = ast.literal_eval(comments)
        if comments and isinstance(comments, list):
            clean_comments = [clean_html(c.replace("\r", " ").replace("\n", " ")) for c in comments[:2]]
            summary.append(f"방문객들의 후기로는 '{' / '.join(clean_comments)}' 등이 있습니다.")
        else:
            summary.append("이용자 후기는 등록되어 있지 않습니다.")
    except:
        summary.append("이용자 후기는 등록되어 있지 않습니다.")

    difficulty = data.get("오지난이도", "")
    if difficulty and is_meaningful(difficulty):
        summary.append(f"오지 난이도는 '{difficulty}' 정도로 평가됩니다.")
    else:
        summary.append("오지 난이도 정보는 따로 확인되지 않았습니다.")

    return " ".join(summary)


def main():
    input_path = Path("campdata.json")
    output_path = Path("camp_data_rag_ready_natural.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data_list = json.load(f)

    results = []
    for item in data_list:
        try:
            print(f"처리중: {item.get('캠핑장이름', '이름없음')}")
            summary = generate_natural_metadata(item)
            results.append({
                "자연어요약": summary,
                "원본데이터": item
            })
        except Exception as e:
            print(f"❌ 요약 실패: {item.get('캠핑장이름', '이름없음')} → {e}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 총 {len(results)}개 캠핑장 요약 완료 → {output_path}")

if __name__ == "__main__":
    main()