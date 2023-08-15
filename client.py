""" Jina Sample Client
    Written by Jacob Barhak for "an evening of python coding" demo
"""
from jina import Client, DocumentArray
import os

docs = DocumentArray.from_files('images' + os.sep + '*.*')

c = Client(port=12345)
r = c.post('/', docs)
for text in r.texts:
    print(text)
