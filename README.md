# Clustering-Based Mathematical Information Retrieval (MathIR) System Backend
This project implements a scalable and efficient Mathematical Information Retrieval (MIR) system using clustering techniques. It focuses on the retrieval of mathematical formulae from large-scale datasets by applying bit-vector-based encoding, unsupervised clustering, and fast query processing. The backend uses FastAPI to make scalable. 

🔧 Tech Stack

FastAPI for API framework

Scikit-learn for MiniBatchKMeans clustering

Python standard libraries

CORS middleware for cross-origin frontend-backend communication

📁 Project Structure
<pre lang="md">
📦backend
 ┣ 📂MIR_model
 ┃ ┣ 📜cluster_index.py          # Handles cluster index loading and searching
 ┃ ┣ 📜clustering_phase.py       # perfoms clustering on bitvector
 ┃ ┣ 📜driver_clustering.py      # Triggers clustering and index creation
 ┃ ┣ 📜driver_preprocessing.py   # Triggers preprocessing of html document and return dictionary of bitvector and correponding metadata
 ┃ ┣ 📜hamming_mini_batch_kmeans.py   # Adapted minibatch kmeans for binary bitvectors
 ┃ ┣ 📜preprocessing.py          # Perform extraction of mathml and latex from html file and generate bitvectors.
 ┃ ┣ 📜query_processing.py       # first check the query for latex or plaintext, if plain text then generate latex.
 ┃ ┣ 📜query_to_bitvector.py     # convert query latex into mathml then mathml into bitvector bitvector generation.
 ┃ ┣ 📜search_query.py           # take user query and triggers searching
 ┣ 📜main.py                     # FastAPI entry point
 ┗ 📁math_index_storage          # Stores saved cluster models and indices and hamming minibatchkmeans model state
</pre>

🚀 How It Works

Clustering & Indexing

On first run, mathematical expressions (e.g., LaTeX/MathML) are vectorized to binary format.

MiniBatchKMeans is applied using Hamming distance to cluster expressions.

Cluster indices are saved to disk.

Searching

User submits a query via the /search endpoint.

The system identifies the nearest cluster(s) and returns top results.

Matching files are returned from the index with minimal latency.

⚙️ API Endpoints

POST /search

Search for math expressions based on query.

Request Body (JSON)

{
  "query": "\\frac{a}{b} + c^2"
}

Response

{
  "time_taken_in_second": 0.25,
  "results": [
    { "id": "1", "filename": "doc1.html" },
    { "id": "2", "filename": "doc2.html" }
  ]
}

🔸 Returns top 20 matching results using cluster-based approximate nearest neighbor (ANN) search.

Error Response (Invalid Input)

{
  "detail": "Invalid input: not a recognizable LaTeX or plain mathematical expression."
}

GET /view/{file_id}

Retrieve HTML content for the result file.

Example

/view/1 → returns the file (e.g., doc1.html) from main memory if present.

Response

Returns full HTML file as plain content.

Errors

404 if file ID not found or file missing from main memory.

🔄 Startup Behavior

On startup, the backend does the following:

Checks for math_index_storage directory.

If not present, runs clustering_and_indexing() to build clusters.

Else, loads MathClusterIndex from main memory.

🔐 CORS Middleware

Allows any frontend (web or mobile) to access the backend:

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

🔸 In production, replace "*" with your domain (e.g., https://yourapp.com).

🧪 Local Development

uvicorn main:app --reload

Server runs on http://127.0.0.1:8000

Visit http://127.0.0.1:8000/docs for Swagger UI

📦 Deployment Notes

Avoid pushing large index files (>50MB) to GitHub.

Use .gitignore to exclude math_index_storage.

## ✉ Contact

**Author:** Ankit Kumar
**Email:** [ankit.kumar@aus.ac.in](mailto:ankit.kumar@aus.ac.in)


## 🚫 License and Usage

This repository is intended for **demonstration purposes only**.  
All source code is **copyright © 2025 Ankit Kumar. All rights reserved.**

You may:
- ✅ View the code and access the demo via provided link.

You may **not**:
- ❌ Copy, clone, or reuse the code.
- ❌ Modify or distribute any part of this project.

Violations may result in legal action under copyright law.



