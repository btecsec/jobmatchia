"""Tokenization réelle d'extraits de CV — Chapitre 2.

Utilise le tokenizer BPE de GPT-2 (via Hugging Face) pour rendre visibles :
- le découpage en sous-mots,
- le surcoût du français par rapport à l'anglais,
- la conversion finale en identifiants numériques.

Règle d'or JobMatch AI : uniquement des textes synthétiques et anonymisés.
"""

import sys

from transformers import AutoTokenizer

# Id canonique du dépôt Hugging Face ("gpt2" court n'est plus accepté
# par les versions récentes de transformers).
MODEL_ID = "openai-community/gpt2"


def load_tokenizer():
    """Charge le tokenizer BPE de GPT-2 (vocabulaire de 50 257 tokens).

    Premier lancement : téléchargement de quelques Mo depuis Hugging Face,
    puis mise en cache locale. Le mode dégradé explicite en cas d'échec
    réseau est un réflexe de production (proxys d'entreprise, hors-ligne) :
    un outil qui explique pourquoi il ne peut pas travailler vaut mieux
    qu'une stack trace de dix écrans.
    """
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID)
    except OSError as exc:
        sys.exit(
            f"Impossible de télécharger le tokenizer '{MODEL_ID}'.\n"
            "Cause probable : pas d'accès réseau à huggingface.co "
            "(proxy, pare-feu, hors-ligne).\n"
            "Réessayez connecté ; le fichier sera ensuite en cache local.\n"
            f"Détail : {exc}"
        )


def inspect(text: str, tokenizer) -> dict:
    """Retourne le détail de tokenization d'un texte : morceaux, ids, comptage."""
    token_ids = tokenizer.encode(text)
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    return {
        "text": text,
        "pieces": pieces,
        "ids": token_ids,
        "n_tokens": len(token_ids),
        "n_chars": len(text),
    }


def demo() -> None:
    tokenizer = load_tokenizer()
    samples = [
        # même sens, deux langues : comparez les comptages
        "Développeur Python, 5 ans d'expérience en backend",
        "Python developer, 5 years of backend experience",
        # mot hors vocabulaire : recomposé en briques BPE
        "Expert JobMatchAI et kubectl",
    ]
    for text in samples:
        info = inspect(text, tokenizer)
        print(f"\n« {info['text']} »")
        print(f"  {info['n_chars']} caractères → {info['n_tokens']} tokens")
        print(f"  morceaux : {info['pieces']}")
        print(f"  ids      : {info['ids']}")


if __name__ == "__main__":
    demo()
