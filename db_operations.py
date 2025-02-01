import psycopg2
from db_pool import get_db_connection, release_db_connection

# 데이터 삽입 함수
def insert_frame(name, age, email):
    conn = get_db_connection()
    if not conn:
        print("❌ DB 연결을 가져올 수 없습니다.")
        return

    try:
        cur = conn.cursor()
        insert_query = "INSERT INTO users (name, age, email) VALUES (%s, %s, %s);"
        cur.execute(insert_query, (name, age, email))
        conn.commit()
        print(f"✅ 사용자 {name} 추가 완료!")
    except psycopg2.Error as e:
        print(f"❌ 데이터 삽입 실패: {e}")
    finally:
        cur.close()
        release_db_connection(conn)  # 연결 반환
