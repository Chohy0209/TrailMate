from neo4j import GraphDatabase
import json
from FlagEmbedding import BGEM3FlagModel
import hashlib

# ===== BGE-M3 임베딩 =====
class UnifiedBGEM3Embedder:
    def __init__(self, model_name="dragonkue/BGE-m3-ko"):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
    
    def encode(self, texts):
        return self.model.encode(texts, batch_size=12, return_dense=True)['dense_vecs']

# ===== Neo4j 연결 =====
uri = "myuri"
driver = GraphDatabase.driver(uri, auth=("neo4j", "mypass"))

# ===== JSON 로딩 =====
with open("camp_data_sentence_with_latlng.json", "r", encoding="utf-8") as f:
    camping_data_list = json.load(f)

embedder = UnifiedBGEM3Embedder()

def generate_unique_id(camp_id, content, node_type):
    """캠프별 고유한 노드 ID 생성"""
    combined = f"{camp_id}_{node_type}_{content}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

# # ===== 기존 데이터 삭제 =====
# with driver.session() as session:
#     print("기존 데이터 삭제 중...")
#     session.run("MATCH (n) DETACH DELETE n")
#     print("기존 데이터 삭제 완료!")

# ===== 새로운 데이터 임베딩 및 저장 =====
with driver.session() as session:
    total_camps = len(camping_data_list)
    
    for idx, camp in enumerate(camping_data_list, 1):
        camp_meta = camp['원본데이터']
        camp_summary_list = camp['자연어요약문장들']

        # Camp 노드 ID - 캠핑장_등록번호로 고유 식별
        camp_registration_no = camp_meta.get('캠핑장_등록번호')
        if not camp_registration_no:
            print(f"캠핑장 등록번호가 없는 데이터 스킵: {camp_meta.get('캠핑장이름', 'Unknown')}")
            continue
            
        camp_node_id = f"camp_{camp_registration_no}"
        camp_vec = embedder.encode([camp_meta.get("캠핑장이름", "")])[0].tolist()

        # =========================
        # Camp 노드 생성
        # =========================
        session.run(
            """
            CREATE (c:Camp {
                registration_no: $registration_no,
                id: $id,
                name: $name,
                status: $status,
                address: $address,
                type: $type,
                facilities: $facilities,
                activities: $activities,
                embedding: $embedding,
                meta: $meta
            })
            """,
            registration_no=camp_registration_no,
            id=camp_node_id,
            name=camp_meta.get("캠핑장이름", ""),
            status=camp_meta.get("운영상태"),
            address=camp_meta.get("캠핑장주소"),
            type=camp_meta.get("캠핑유형"),
            facilities=camp_meta.get("캠핑장시설"),
            activities=camp_meta.get("즐길거리"),
            embedding=camp_vec,
            meta=json.dumps(camp_meta, ensure_ascii=False)
        )

        # =========================
        # Attribute 노드 생성 및 연결
        # =========================
        attributes = [
            ("장소유형", camp_meta.get("캠핑장_장소유형")),
            ("지형특성", camp_meta.get("지형특성")),
            ("가격", camp_meta.get("가격")),
            ("홈페이지", camp_meta.get("홈페이지")),
            ("예약주소", camp_meta.get("예약주소")),
            ("후기", camp_meta.get("캠핑장댓글")),
            ("메모", camp_meta.get("캠핑장메모")),
            ("설명", camp_meta.get("캠핑장설명")),
            ("연락처", camp_meta.get("연락처")),
        ]

        for i in range(1, 4):
            title = camp_meta.get(f"추가제목{i}")
            content = camp_meta.get(f"추가내용{i}")
            if title and content and title.lower() != "정보없음" and content.lower() != "정보없음":
                attributes.append((title, content))

        for attr_name, attr_value in attributes:
            if attr_value:
                attr_id = generate_unique_id(camp_node_id, str(attr_value), attr_name)
                attr_vec = embedder.encode([str(attr_value)])[0].tolist()
                
                session.run(
                    """
                    MATCH (c:Camp {registration_no: $registration_no})
                    CREATE (a:Attribute {
                        id: $attr_id,
                        camp_id: $camp_id,
                        type: $type,
                        text: $text,
                        embedding: $embedding
                    })
                    CREATE (c)-[:HAS_ATTRIBUTE {type: $type}]->(a)
                    """,
                    registration_no=camp_registration_no,
                    attr_id=attr_id,
                    camp_id=camp_node_id,
                    text=str(attr_value),
                    embedding=attr_vec,
                    type=attr_name
                )

        # =========================
        # Summary 노드 생성 및 연결
        # =========================
        for sentence in camp_summary_list:
            summary_id = generate_unique_id(camp_node_id, sentence, "summary")
            sent_vec = embedder.encode([sentence])[0].tolist()
            
            session.run(
                """
                MATCH (c:Camp {registration_no: $registration_no})
                CREATE (s:Summary {
                    id: $summary_id,
                    camp_id: $camp_id,
                    text: $text,
                    embedding: $embedding
                })
                CREATE (c)-[:HAS_SUMMARY]->(s)
                """,
                registration_no=camp_registration_no,
                summary_id=summary_id,
                camp_id=camp_node_id,
                text=sentence,
                embedding=sent_vec
            )

        print(f"[{idx}/{total_camps}] 캠프 '{camp_meta.get('캠핑장이름', 'Unknown')}' 처리 완료")

print("모든 캠프 데이터 Neo4j에 새로 저장 완료!")
