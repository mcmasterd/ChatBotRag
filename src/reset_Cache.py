import redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, password='terminator')
r.flushdb()
print("Đã xóa toàn bộ cache Redis!")