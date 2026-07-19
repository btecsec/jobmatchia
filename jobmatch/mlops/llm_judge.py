"""LLM-as-a-Judge — Chapitre 8 : évaluer ce qui n'a pas de métrique.

Le rappel et le F1 jugent un classifieur. Mais quelle métrique juge une
lettre de motivation générée ? Réponse de l'industrie : un second
modèle, le « juge », qui compare deux candidats selon une grille
explicite et rend un verdict structuré.

Ce module implémente le PROTOCOLE complet du juge — grille, verdict
JSON strict, anti-biais de position par double passage — avec une
interface unique : `judge_fn(prompt) -> str`. En production, judge_fn
appelle un LLM (OpenRouter, par exemple). Dans les tests et la démo,
un juge heuristique déterministe joue le rôle : le protocole testé est
exactement celui qui encadrera le vrai LLM.

Règle d'or : le juge ne voit que des textes déjà anonymisés (Ch.4).
Sa grille contient d'ailleurs un critère « aucune donnée personnelle »
— le juge est une couche de défense de PLUS, pas un substitut à la
barrière déterministe.
"""

import json
import re
from collections.abc import Callable

RUBRIC = """Tu es le juge qualité de JobMatch AI. Compare deux lettres de
motivation pour la même offre, selon ces critères par ordre d'importance :
1. Les compétences requises par l'offre sont-elles mises en avant ?
2. La lettre est-elle exempte de toute donnée personnelle (email, téléphone) ?
3. La longueur est-elle raisonnable (50 à 200 mots) ?
Réponds UNIQUEMENT en JSON : {"winner": "A" | "B" | "tie", "reason": "..."}"""

_LETTER_A = "=== LETTRE A ==="
_LETTER_B = "=== LETTRE B ==="
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def build_judge_prompt(offer_skills: list[str], letter_a: str,
                       letter_b: str) -> str:
    """Assemble le prompt du juge : grille, offre, les deux candidates."""
    return (f"{RUBRIC}\n\nCompétences requises par l'offre : "
            f"{', '.join(offer_skills)}\n\n"
            f"{_LETTER_A}\n{letter_a}\n\n{_LETTER_B}\n{letter_b}")


def parse_verdict(raw: str) -> str:
    """Verdict JSON strict : tout ce qui n'est pas conforme est rejeté.

    Un LLM bavard qui enrobe son JSON de politesses, ou qui invente un
    gagnant « C », doit faire échouer le parsing — jamais être deviné.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"verdict illisible : {raw[:80]!r}") from exc
    winner = data.get("winner")
    if winner not in ("A", "B", "tie"):
        raise ValueError(f"gagnant invalide : {winner!r}")
    return winner


def judge_pair(judge_fn: Callable[[str], str], offer_skills: list[str],
               letter_a: str, letter_b: str) -> str:
    """Deux passages, ordres inversés : l'anti-biais de position.

    Les LLM juges favorisent statistiquement la première réponse
    présentée. Parade standard : juger (A, B) puis (B, A) ; si les deux
    verdicts ne désignent pas le même texte, le duel est déclaré nul —
    un juge inconstant n'a pas voix au chapitre.
    """
    first = parse_verdict(judge_fn(
        build_judge_prompt(offer_skills, letter_a, letter_b)))
    second = parse_verdict(judge_fn(
        build_judge_prompt(offer_skills, letter_b, letter_a)))
    second_unswapped = {"A": "B", "B": "A", "tie": "tie"}[second]
    return first if first == second_unswapped else "tie"


def heuristic_judge(prompt: str) -> str:
    """Juge déterministe : applique la grille à la lettre, sans LLM.

    Il tient le rôle du LLM dans les tests — même entrée (le prompt
    complet), même sortie (un JSON de verdict). Le jour où un vrai
    modèle le remplace, rien d'autre ne change dans le pipeline.
    """
    skills_line = re.search(r"Compétences requises par l'offre : (.+)",
                            prompt)
    skills = [s.strip() for s in skills_line.group(1).split(",")]
    letter_a = prompt.split(_LETTER_A)[1].split(_LETTER_B)[0].strip()
    letter_b = prompt.split(_LETTER_B)[1].strip()

    def score(letter: str) -> int:
        lowered = letter.lower()
        points = sum(2 for skill in skills if skill.lower() in lowered)
        if _EMAIL_RE.search(letter):
            points -= 10
        if 50 <= len(letter.split()) <= 200:
            points += 1
        return points

    score_a, score_b = score(letter_a), score(letter_b)
    winner = "A" if score_a > score_b else "B" if score_b > score_a else "tie"
    return json.dumps({"winner": winner,
                       "reason": f"scores grille A={score_a}, B={score_b}"})


def position_biased_judge(prompt: str) -> str:
    """Le juge défaillant du chapitre : il préfère toujours la position A.

    Sert à démontrer que le double passage neutralise ce biais.
    """
    return json.dumps({"winner": "A", "reason": "la première me semble mieux"})


def demo() -> None:
    """Un duel de lettres, un juge honnête, un juge biaisé — et le protocole."""
    offer = ["docker", "kubernetes", "python"]
    good = ("Votre offre a retenu toute mon attention : cinq années à "
            "automatiser des déploiements Docker et Kubernetes, et un "
            "quotidien outillé en Python, correspondent précisément aux "
            "compétences que vous recherchez. Je serais ravi d'échanger "
            "sur vos enjeux d'infrastructure et d'apporter à votre équipe "
            "des pipelines fiables, mesurés et documentés.")
    weak = ("Je suis très motivé et je m'adapte vite. "
            "Écrivez-moi sur alex.martin@exemple.fr pour en discuter.")

    print("Duel : lettre A (ciblée) contre lettre B (générique + fuite email)")
    verdict = judge_pair(heuristic_judge, offer, good, weak)
    print(f"  Juge honnête, double passage : gagnant = {verdict}")

    verdict = judge_pair(position_biased_judge, offer, good, weak)
    print(f"  Juge biaisé-position, double passage : verdict = {verdict} "
          f"(le protocole neutralise le biais)")

    single = parse_verdict(position_biased_judge(
        build_judge_prompt(offer, weak, good)))
    print(f"  Le même juge biaisé, SANS double passage : {single} "
          f"— la lettre faible aurait gagné")


if __name__ == "__main__":
    demo()
