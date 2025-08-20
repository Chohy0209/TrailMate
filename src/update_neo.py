def update_camp_embeddings_with_full_info(driver, embedder):
    """기존 Camp 노드들의 임베딩을 풍부한 정보로 재생성"""
    
    with driver.session() as session:
        # 1. 모든 Camp 노드 조회
        result = session.run("""
            MATCH (c:Camp)
            RETURN c.id as camp_id, c.name as name, c.address as address, 
                   c.type as type, c.facilities as facilities, c.activities as activities,
                   c.meta as meta
        """)
        
        camps = list(result)
        total_camps = len(camps)
        print(f"총 {total_camps}개 캠프 노드의 임베딩을 업데이트합니다...")
        
        for idx, record in enumerate(camps, 1):
            try:
                # 2. 기본 정보만으로 텍스트 구성
                name = record["name"] or ""
                address = record["address"] or ""
                camp_type = record["type"] or ""
                facilities = record["facilities"] or ""
                activities = record["activities"] or ""
                
                # 3. 기본 정보만으로 임베딩 생성
                full_text = f"""
                캠핑장: {name}
                주소: {address}
                유형: {camp_type}
                시설: {facilities}
                활동: {activities}
                """.strip()
                
                # 4. 새 임베딩 생성
                new_embedding = embedder.encode([full_text])[0].tolist()
                
                # 5. 임베딩 업데이트
                session.run("""
                    MATCH (c:Camp {id: $camp_id})
                    SET c.embedding = $new_embedding
                """, 
                camp_id=record["camp_id"],
                new_embedding=new_embedding
                )
                
                if idx % 50 == 0:
                    print(f"진행률: {idx}/{total_camps} ({idx/total_camps*100:.1f}%)")
                    
            except Exception as e:
                print(f"오류 발생 (캠프 {record['camp_id']}): {e}")
                continue
        
        print(f"✅ {total_camps}개 캠프 임베딩 업데이트 완료!")
        
        # 6. 벡터 인덱스 재구축 (선택사항)
        print("벡터 인덱스 재구축 중...")
        try:
            session.run("DROP INDEX camp_embedding_index IF EXISTS")
            session.run(f"""
                CREATE VECTOR INDEX camp_embedding_index
                FOR (c:Camp) ON (c.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {VECTOR_DIM},
                    `vector.similarity_function`: '{SIM_FUNC}'
                  }}
                }}
            """)
            print("✅ 벡터 인덱스 재구축 완료!")
        except Exception as e:
            print(f"인덱스 재구축 중 오류: {e}")

# 실행
update_camp_embeddings_with_full_info(driver, embedder)