from pymongo import MongoClient

uri = "mongodb+srv://studyadmin:kushant%4012345@cluster0.z6cxhvn.mongodb.net/?appName=Cluster0"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    print(client.admin.command("ping"))
    print("CONNECTED SUCCESSFULLY")
except Exception as e:
    print("ERROR:")
    print(e)