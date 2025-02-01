import psycopg2
from psycopg2 import pool

# PostgreSQL 연결 정보
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "mydatabase",
    "user": "myuser",
    "password": "mypassword",
}

# Connection Pool 생성
try:
    db_pool = pool.SimpleConnectionPool(
        minconn=1,  # 최소 연결 개수
        maxconn=5,  # 최대 연결 개수
        **DB_CONFIG
    )
    print("✅ Connection Pool 생성 완료!")
except psycopg2.Error as e:
    print(f"❌ DB 연결 오류 발생: {e}")
    db_pool = None

# 커넥션 가져오기
def get_db_connection():
    if db_pool:
        return db_pool.getconn()
    return None

# 커넥션 반환
def release_db_connection(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

# Connection Pool 닫기 (필요할 때 호출)
def close_connection_pool():
    if db_pool:
        db_pool.closeall()
        print("✅ 모든 DB 연결 닫힘!")