from rag import Rag
from evaluator import RagEvaluator
open_ai = Rag('fixed_size_1500', 'GPT-3.5')

rag_evaluator = RagEvaluator(open_ai)

# rag_evaluator.get_golden_dataset(1)

result = rag_evaluator.get_mean_result()


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

