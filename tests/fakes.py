class FakeRedis:
    def __init__(self, decode_responses=True):
        self.decode_responses = decode_responses
        self._strings = {}
        self._hashes = {}
        self._sorted_sets = {}
        self._hyperloglogs = {}

    def zadd(self, key, mapping):
        bucket = self._sorted_sets.setdefault(key, {})
        for member, score in mapping.items():
            bucket[member] = float(score)

    def zremrangebyscore(self, key, min_score, max_score):
        bucket = self._sorted_sets.get(key, {})
        lower = float("-inf") if min_score == "-inf" else float(min_score)
        upper = float(max_score)
        members_to_remove = [member for member, score in bucket.items() if lower <= score <= upper]
        for member in members_to_remove:
            del bucket[member]

    def zrangebyscore(self, key, min_score, max_score):
        bucket = self._sorted_sets.get(key, {})
        lower = float(min_score)
        upper = float(max_score)
        members = [(member, score) for member, score in bucket.items() if lower <= score <= upper]
        members.sort(key=lambda item: (item[1], item[0]))
        return [member for member, _ in members]

    def pfadd(self, key, *values):
        bucket = self._hyperloglogs.setdefault(key, set())
        before = len(bucket)
        bucket.update(values)
        return 1 if len(bucket) > before else 0

    def pfcount(self, key):
        return len(self._hyperloglogs.get(key, set()))

    def hgetall(self, key):
        return dict(self._hashes.get(key, {}))

    def hset(self, key, mapping):
        bucket = self._hashes.setdefault(key, {})
        for field, value in mapping.items():
            bucket[field] = str(value)

    def get(self, key):
        return self._strings.get(key)

    def set(self, key, value):
        self._strings[key] = str(value)

    def setex(self, key, ttl, value):
        self.set(key, value)

    def expire(self, key, ttl):
        return True
