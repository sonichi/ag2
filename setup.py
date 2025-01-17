# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2labs
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
import platform
import sys

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# Get the code version
version = {}
with open(os.path.join(here, "autogen/version.py")) as fp:
    exec(fp.read(), version)
__version__ = version["__version__"]


current_os = platform.system()

install_requires = [
    "openai>=1.3",
    "diskcache",
    "termcolor",
    "flaml",
    # numpy is installed by flaml, but we want to pin the version to below 2.x (see https://github.com/microsoft/autogen/issues/1960)
    "numpy>=1.17.0,<2",
    "python-dotenv",
    "tiktoken",
    # Disallowing 2.6.0 can be removed when this is fixed https://github.com/pydantic/pydantic/issues/8705
    "pydantic>=1.10,<3,!=2.6.0",  # could be both V1 and V2
    "docker",
    "packaging",
]

jupyter_executor = [
    "jupyter-kernel-gateway",
    "websocket-client",
    "requests",
    "jupyter-client>=8.6.0",
    "ipykernel>=6.29.0",
]

retrieve_chat = [
    "protobuf==4.25.3",
    "chromadb==0.5.3",
    "sentence_transformers",
    "pypdf",
    "ipython",
    "beautifulsoup4",
    "markdownify",
]

retrieve_chat_pgvector = [*retrieve_chat, "pgvector>=0.2.5"]

graph_rag_falkor_db = [
    "graphrag_sdk",
]

if current_os in ["Windows", "Darwin"]:
    retrieve_chat_pgvector.extend(["psycopg[binary]>=3.1.18"])
elif current_os == "Linux":
    retrieve_chat_pgvector.extend(["psycopg>=3.1.18"])

extra_require = {
    "test": [
        "ipykernel",
        "nbconvert",
        "nbformat",
        "pre-commit",
        "pytest-cov>=5",
        "pytest-asyncio",
        "pytest>=6.1.1,<8",
        "pandas",
    ],
    "blendsearch": ["flaml[blendsearch]"],
    "mathchat": ["sympy", "pydantic==1.10.9", "wolframalpha"],
    "retrievechat": retrieve_chat,
    "retrievechat-pgvector": retrieve_chat_pgvector,
    "retrievechat-mongodb": [*retrieve_chat, "pymongo>=4.0.0"],
    "retrievechat-qdrant": [*retrieve_chat, "qdrant_client", "fastembed>=0.3.1"],
    "graph_rag_falkor_db": graph_rag_falkor_db,
    "autobuild": ["chromadb", "sentence-transformers", "huggingface-hub", "pysqlite3"],
    "teachable": ["chromadb"],
    "lmm": ["replicate", "pillow"],
    "graph": ["networkx", "matplotlib"],
    "gemini": ["google-generativeai>=0.5,<1", "google-cloud-aiplatform", "google-auth", "pillow", "pydantic"],
    "together": ["together>=1.2"],
    "websurfer": ["beautifulsoup4", "markdownify", "pdfminer.six", "pathvalidate"],
    "redis": ["redis"],
    "cosmosdb": ["azure-cosmos>=4.2.0"],
    "websockets": ["websockets>=12.0,<13"],
    "jupyter-executor": jupyter_executor,
    "types": ["mypy==1.9.0", "pytest>=6.1.1,<8"] + jupyter_executor,
    "long-context": ["llmlingua<0.3"],
    "anthropic": ["anthropic>=0.23.1"],
    "cerebras": ["cerebras_cloud_sdk>=1.0.0"],
    "mistral": ["mistralai>=1.0.1"],
    "groq": ["groq>=0.9.0"],
    "cohere": ["cohere>=5.5.8"],
    "ollama": ["ollama>=0.3.3", "fix_busted_json>=0.0.18"],
    "bedrock": ["boto3>=1.34.149"],
}


if "--name" in sys.argv:
    index = sys.argv.index("--name")
    sys.argv.pop(index)  # Removes --name
    package_name = sys.argv.pop(index)  # Removes the value after --name
else:
    package_name = "autogen"


setuptools.setup(
    name=package_name,
    version=__version__,
    author="Chi Wang & Qingyun Wu",
    author_email="auto-gen@outlook.com",
    description="A programming framework for agentic AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ag2labs/ag2",
    packages=setuptools.find_packages(include=["autogen*"], exclude=["test"]),
    install_requires=install_requires,
    extras_require=extra_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache Software License 2.0",
    python_requires=">=3.8,<3.13",
)
