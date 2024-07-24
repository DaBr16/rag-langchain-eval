from rag import Rag
from evaluator import RagEvaluator
recursive_2000_rag = Rag(chunkingstrategy='recursive_size_2000', llm='gpt-3.5-turbo-0125', embedding='text-embedding-3-small')
recursive_2000_eval = RagEvaluator(rag=recursive_2000_rag, eval_llm='gpt-3.5-turbo-0125', embedding_function='text-embedding-3-small', num_of_runs=1, num_of_questions=2)


recursive_2000_eval.get_golden_dataset(1)

result = recursive_2000_eval.get_mean_result()




print(result)

# One Run
# {'answer_relevancy': 0.3316, 'answer_correctness': 0.3489, 'context_recall': 0.3296, 'context_relevancy': 0.0168}

# Five Runs
# {'answer_relevancy': 0.3661, 'answer_correctness': 0.2709, 'context_recall': 0.3374, 'context_relevancy': 0.0327}


# offene Fragen
# Wie richtig ist der Sentence Splitter (z.B. Wird "U.S."" nicht richtig erkannt)
# Wie sehen die Chunks nun aus? (Inhalt, Größe, Anzahl) Da ich immer nur 1000 Sätze nehmen und daraus die Chunks generiere.
# Wie würde der Semantic Chunker funktionieren, wenn einfach die Sätze mit den ähnlichsten Embeddings zusammen gewürfelt werden? 
# (Würde sich was ändern wenn die Daten anders angeordnet wären?)
# Wie ist das Ergebnis, wenn wir nach Standardabweichung oder Interquartilsabstand anstatt Prozentwert abspalten?

