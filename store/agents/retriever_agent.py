from modules.ir import SimpleIR
from modules.security import sanitize_text

class RetrieverAgent:
    def __init__(self, data_path: str = None):
        self.ir = SimpleIR()
        self.ir.load_data(data_path)
        self.ir.build_index()

    def handle_query(self, query_text: str, top_k: int = 5):
        q = sanitize_text(query_text)
        results = self.ir.query(q, top_k=top_k)
        # return only essential fields for privacy
        cleaned = []
        for r in results:
            row = r['row']
            # keep score and a short preview
            preview = (row.get('text') or '')[:300]
            cleaned.append({'score': r['score'], 'preview': preview, 'index': r['index']})
        return cleaned
