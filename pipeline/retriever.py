from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

############################################## BM25 Retriever ##############################################

class BM25_retriever():
    def __init__(self, target_chunk_size=250, overlap=40, n=10):
        '''
        Args:
            target_chunk_size (int): The target size of each chunk (in words).
            overlap (int): The number of overlapping words between chunks.
            n (int): Number of top chunks to retrieve across all documents.
        '''
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap
        self.n = n

    def split_into_chunks(self, text):
        """
        Split a document into chunks with complete sentences and rough size constraints.
        Args:
            text (str): The document text to split.
        Returns:
            list: List of text chunks.
        """
        # split into sentences using nltk
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        # Chunk grouping while respecting sentence boundaries
        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            # If adding this sentence exceeds the target chunk size (and the chunk is not empty), finalize the chunk
            if current_word_count + sentence_word_count > self.target_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Start a new chunk with overlap
                overlap_words = " ".join(current_chunk[-self.overlap:]).split() if self.overlap > 0 else []
                current_chunk = overlap_words + [sentence]
                current_word_count = len(overlap_words) + sentence_word_count
            else:
                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_word_count

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def retrieve(self, claim, evidence_list):
        """
        Retrieve relevant chunks from a combined document, while tracking which original document each chunk belongs to.
        Args:
            claim (str): The claim to match against.
            evidence_list (list): List of evidence dictionaries (each containing 'text').
        Returns:
            list: List of dictionaries with 'title', 'description', and 'retrieved_text' for each chunk.
        """
        # Combine all evidence documents into a single text corpus, while tracking source information
        combined_chunks = []
        for ev in evidence_list:
            if 'text' in ev and ev['text'].strip():
                # use crawled text from webpages, remove empty lines
                text_to_add = "\n".join(line for line in ev['text'].splitlines() if line.strip())
            elif 'description' in ev:
                text_to_add = ev['description']
            else:
                continue  # Skip if no text or description is available

            # Split the document into chunks and track source information
            chunks = self.split_into_chunks(text_to_add)
            for chunk in chunks:
                if chunk.strip():
                    combined_chunks.append({
                        'title': ev.get('title', ''),
                        'description': ev.get('description', ''),
                        'text': chunk
                    })

        # Tokenize the claim and chunks
        tokenized_chunks = []
        for info in combined_chunks:
            # Use scraped text if available; fallback to description if not (with final empty check)
            text_to_tokenize = info['text'].strip() or info['description'].strip()
            tokenized = text_to_tokenize.split() if text_to_tokenize else []
            tokenized_chunks.append(tokenized)
        tokenized_claim = claim.split()

        try:
            bm25 = BM25Okapi(tokenized_chunks)
            # Get the top n relevant chunks for the claim
            top_indices = bm25.get_top_n(tokenized_claim, range(len(combined_chunks)), n=self.n)
        except Exception as e:
            print(f"Error during BM25 processing: {str(e)}")
            # TODO: search web again if detected such cases
            return combined_chunks

        # Reorganize the top chunks
        retrieved_chunks = []
        for idx in top_indices:
            retrieved_chunks.append({
                'title': combined_chunks[idx]['title'], # the title of the original document that the chunk comes from
                'description': combined_chunks[idx]['description'], # the description of the original document that the chunk comes from
                'retrieved_text': combined_chunks[idx]['text']
            })

        return retrieved_chunks