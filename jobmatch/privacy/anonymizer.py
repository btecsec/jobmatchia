"""Anonymiseur déterministe de JobMatch AI — Chapitre 4.

La règle d'or du projet : aucun agent IA ne reçoit ni ne traite jamais de
données d'identification directe (nom, prénom, âge, adresse, coordonnées).

Ce module est la première moitié de cette garantie. Il est volontairement
NON-IA : uniquement des remplacements exacts et des expressions régulières,
donc un comportement prouvable, testable et auditable. Un modèle de ML
« détecteur de PII » aurait un taux d'erreur ; ici, une PII connue est
retirée à 100 %, ou le test échoue.

Architecture (cf. schéma du livre) :

    texte brut 🔴 ──> DeterministicAnonymizer ──> texte anonymisé 🟢 + task_id
                              │
                              └──> PiiVault (coffre PII, hors zone IA)

La seconde moitié — le module enrichisseur qui réinjecte les PII en toute
fin de pipeline — est construite au Chapitre 14 ; `enrich()` en donne ici
l'aperçu minimal pour prouver l'aller-retour sans perte.
"""

import re
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    """Les PII connues de l'utilisateur, déclarées à l'inscription.

    C'est le grand avantage architectural de JobMatch AI : l'application
    SAIT qui est l'utilisateur. Pas besoin de deviner son nom par IA —
    il suffit de retirer des valeurs connues (déterministe par nature).
    """

    prenom: str
    nom: str
    email: str
    telephone: str
    adresse: str = ""
    date_naissance: str = ""  # format JJ/MM/AAAA


# Filet de sécurité : PII que l'utilisateur pourrait avoir écrites dans son
# CV sous une autre forme que celle déclarée (second email, autre numéro...).
# L'ordre compte : les motifs les plus spécifiques d'abord.
SAFETY_NET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("LINKEDIN", re.compile(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+", re.IGNORECASE)),
    ("TEL", re.compile(r"(?:\+33\s?[1-9]|0[1-9])(?:[\s.\-]?\d{2}){4}")),
    ("DATE_NAISSANCE", re.compile(r"\b\d{2}/\d{2}/\d{4}\b")),
    ("ADRESSE", re.compile(
        r"\b\d{1,4}(?:\s(?:bis|ter))?,?\s"
        r"(?:rue|avenue|boulevard|all[ée]e|impasse|chemin|place)\s"
        r"[^,\n]+(?:,\s*\d{5}\s+[\w\-' ]+)?", re.IGNORECASE)),
    # « 32 ans » est une PII (âge), « 10 ans d'expérience » n'en est pas une :
    # le lookahead négatif épargne les années d'expérience. Limite assumée
    # d'une regex — documentée, testée, et renforcée par find_pii_leaks().
    ("AGE", re.compile(r"\b\d{2}\s?ans\b(?!\s+d['e’])")),
]


class PiiVault:
    """Coffre PII : associe un task_id aux correspondances placeholder → valeur.

    Version pédagogique en mémoire. En production (Partie VII) : stockage
    chiffré au repos, hors de portée de la zone IA, purge RGPD planifiée.
    L'important est le CONTRAT : la zone IA ne connaît que le task_id,
    jamais le contenu du coffre.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    def store(self, task_id: str, mapping: dict[str, str]) -> None:
        self._store[task_id] = dict(mapping)

    def retrieve(self, task_id: str) -> dict[str, str]:
        if task_id not in self._store:
            raise KeyError(f"task_id inconnu du coffre : {task_id}")
        return dict(self._store[task_id])


@dataclass(frozen=True)
class AnonymizedDocument:
    """Ce qui sort de l'anonymiseur : du texte 🟢 et une clé opaque.

    C'est TOUT ce que la zone IA recevra. Pas de champ nom, email ou
    téléphone : le schéma lui-même rend la fuite impossible à écrire.
    """

    task_id: str
    text: str


class DeterministicAnonymizer:
    """Retire les PII d'un texte par remplacements exacts + regex.

    Deux couches de défense :
    1. Les valeurs DÉCLARÉES de l'Identity (remplacement exact,
       insensible à la casse) — couverture garantie des PII connues.
    2. Le filet de sécurité regex — attrape les PII écrites sous une
       autre forme (second email, numéro reformaté...).
    """

    def __init__(self, vault: PiiVault) -> None:
        self._vault = vault

    def anonymize(self, text: str, identity: Identity) -> AnonymizedDocument:
        mapping: dict[str, str] = {}          # placeholder -> valeur d'origine
        seen: dict[str, str] = {}             # valeur rencontrée -> placeholder
        counters: dict[str, int] = {}

        def placeholder_for(category: str, matched: str) -> str:
            if matched in seen:
                return seen[matched]
            counters[category] = counters.get(category, 0) + 1
            n = counters[category]
            ph = f"{{{{{category}}}}}" if n == 1 else f"{{{{{category}_{n}}}}}"
            mapping[ph] = matched
            seen[matched] = ph
            return ph

        def substitute(txt: str, category: str, pattern: re.Pattern[str]) -> str:
            return pattern.sub(
                lambda m: placeholder_for(category, m.group(0)), txt)

        # Couche 1 : les valeurs connues, de la plus longue à la plus courte
        # (« Alex Martin » avant « Alex », sinon remplacement partiel).
        # Les noms sont bornés par \b : sans cela, un prénom court comme
        # « Ali » serait remplacé au milieu de « qualité ».
        known: list[tuple[str, str, bool]] = [
            ("NOM_COMPLET", f"{identity.prenom} {identity.nom}", True),
            ("NOM_COMPLET", f"{identity.nom} {identity.prenom}", True),
            ("ADRESSE", identity.adresse, False),
            ("EMAIL", identity.email, False),
            ("TEL", identity.telephone, False),
            ("DATE_NAISSANCE", identity.date_naissance, False),
            ("NOM", identity.nom, True),
            ("PRENOM", identity.prenom, True),
        ]
        for category, value, word_bounded in sorted(
                known, key=lambda kv: len(kv[1]), reverse=True):
            if value:
                escaped = re.escape(value)
                if word_bounded:
                    escaped = rf"\b{escaped}\b"
                text = substitute(
                    text, category, re.compile(escaped, re.IGNORECASE))

        # Couche 2 : le filet de sécurité.
        for category, pattern in SAFETY_NET_PATTERNS:
            text = substitute(text, category, pattern)

        task_id = str(uuid.uuid4())
        self._vault.store(task_id, mapping)
        return AnonymizedDocument(task_id=task_id, text=text)

    def enrich(self, document: AnonymizedDocument) -> str:
        """Réinjecte les PII depuis le coffre (aperçu du module du Ch.14).

        Déterministe lui aussi : de simples remplacements de chaînes.
        Garantie testée : enrich(anonymize(x)) == x.
        """
        text = document.text
        for ph, value in self._vault.retrieve(document.task_id).items():
            text = text.replace(ph, value)
        return text


def find_pii_leaks(text: str, identity: Identity) -> list[str]:
    """Contrat de code de la barrière PII : liste les PII encore présentes.

    Utilisé par les tests de non-régression et, au Chapitre 17, par le
    pentest de la barrière. Doit renvoyer [] pour tout texte qui entre
    en zone IA — sinon le déploiement est bloqué.
    """
    leaks: list[str] = []
    checks = [(identity.prenom, True), (identity.nom, True),
              (identity.email, False), (identity.telephone, False),
              (identity.adresse, False), (identity.date_naissance, False)]
    for value, word_bounded in checks:
        if not value:
            continue
        escaped = re.escape(value)
        if word_bounded:
            escaped = rf"\b{escaped}\b"
        if re.search(escaped, text, re.IGNORECASE):
            leaks.append(value)
    for _category, pattern in SAFETY_NET_PATTERNS:
        leaks.extend(pattern.findall(text))
    return leaks


def demo() -> None:
    """Aller-retour complet : CV brut -> anonymisé -> zone IA -> enrichi."""
    alex = Identity(
        prenom="Alex",
        nom="Martin",
        email="alex.martin@exemple.fr",
        telephone="06 12 34 56 78",
        adresse="12 rue des Lilas, 75011 Paris",
        date_naissance="14/03/1994",
    )
    cv_brut = (
        "Alex Martin — 32 ans\n"
        "12 rue des Lilas, 75011 Paris — alex.martin@exemple.fr — 06 12 34 56 78\n"
        "linkedin.com/in/alex-martin-dev — perso : amartin.pro@exemple.fr\n"
        "\n"
        "Reconversion développeur backend, 10 ans d'expérience en logistique.\n"
        "Compétences : Python, SQL, Docker, API REST.\n"
    )

    vault = PiiVault()
    anonymizer = DeterministicAnonymizer(vault)
    doc = anonymizer.anonymize(cv_brut, alex)

    print("=== CV BRUT (zone applicative uniquement) ===")
    print(cv_brut)
    print("=== CV ANONYMISÉ (seul texte autorisé en zone IA) ===")
    print(doc.text)
    print("=== COFFRE PII (jamais transmis à la zone IA) ===")
    for ph, value in vault.retrieve(doc.task_id).items():
        print(f"  {ph:<22} -> {value}")
    print()
    leaks = find_pii_leaks(doc.text, alex)
    print(f"Fuites PII détectées dans le texte anonymisé : {leaks or 'aucune'}")
    print(f"Aller-retour sans perte : {anonymizer.enrich(doc) == cv_brut}")


if __name__ == "__main__":
    demo()
