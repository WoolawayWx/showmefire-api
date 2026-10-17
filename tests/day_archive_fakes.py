"""In-memory S3 stand-in shared by the day-archive tests."""
import io
import json
from pathlib import Path


class FakeBody(io.BytesIO):
    pass


class FakeS3:
    def __init__(self):
        self.objects = {}

    # writes
    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = bytes(Body)

    def upload_file(self, filename, Bucket, Key, ExtraArgs=None):
        self.objects[Key] = Path(filename).read_bytes()

    def copy(self, source, Bucket, Key):
        self.objects[Key] = self.objects[source["Key"]]

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}

    # reads
    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            error = KeyError(Key)
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        return {"Body": FakeBody(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def download_file(self, Bucket, Key, filename):
        Path(filename).write_bytes(self.objects[Key])

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://r2.example/{Params['Key']}?exp={ExpiresIn}"

    def get_paginator(self, name):
        store = self

        class Paginator:
            def paginate(self, Bucket, Prefix=""):
                keys = sorted(k for k in store.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k, "Size": len(store.objects[k])} for k in keys]}

        return Paginator()

    def json(self, key):
        return json.loads(self.objects[key])
