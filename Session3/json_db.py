import os
from tinydb import TinyDB
from constants import DB_DIR, DB_PATH

if not os.path.isdir(DB_DIR):
    os.makedirs(DB_DIR, exist_ok=True)
db = TinyDB(DB_PATH)


def db_insert(data):
    print("data = ", data)
    db.insert_multiple(data)


def db_get():
    objs = sorted(db.all(), key=lambda k: k['timestamp'])
    if(objs):
        print("objs = ", objs)
        return objs[-6:-1]
    return []
