
import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)

# Définition des rôles possibles pour les messages
Role = Literal["system", "user", "assistant"]

# Structure d'un message dans une conversation
class Message(TypedDict):
    role: Role  # Rôle du message : "system", "user", ou "assistant"
    content: str  # Contenu du message

# Une conversation est une séquence de messages
Dialog = Sequence[Message]

class AvikamTokenizer:
    """
    Tokenizer pour le modèle avikam1-llm.
    Ce tokenizer utilise Tiktoken pour l'encodage et le décodage de texte.
    """

    special_tokens: Dict[str, int]  # Dictionnaire des jetons spéciaux
    num_reserved_special_tokens = 256  # Nombre de jetons réservés pour des usages spécifiques

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initialise le tokenizer avec un modèle Tiktoken.

        Args:
            model_path (str): Chemin vers le fichier du modèle Tiktoken.
        """
        assert os.path.isfile(model_path), f"Fichier modèle non trouvé à {model_path}"

        # Chargement des rangs fusionnables (mergeable ranks) depuis le modèle Tiktoken
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        # Liste des jetons spéciaux utilisés par avikam1-llm
        special_tokens = [
            "<|begin_of_text|>",  # Début du texte
            "<|end_of_text|>",  # Fin du texte
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",  # Début d'un en-tête de message
            "<|end_header_id|>",  # Fin d'un en-tête de message
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # Fin d'un tour de conversation
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]

        # Attribution d'indices aux jetons spéciaux
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        # Initialisation de l'encodeur Tiktoken
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Modèle Tiktoken rechargé depuis {model_path}")

        # Propriétés importantes du tokenizer
        self.n_words: int = self.model.n_vocab  # Taille du vocabulaire
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]  # ID du début de texte
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]  # ID de fin de texte
        self.pad_id: int = -1  # ID de remplissage (padding)
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }  # Jetons de fin de génération
        logger.info(
            f"#mots: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encode une chaîne de caractères en une liste d'IDs de tokens.

        Args:
            s (str): Chaîne à encoder.
            bos (bool): Ajouter le token de début de séquence ?
            eos (bool): Ajouter le token de fin de séquence ?
            allowed_special ("all"|set[str]): Jetons spéciaux autorisés.
            disallowed_special ("all"|set[str]): Jetons spéciaux interdits.

        Returns:
            List[int]: Liste des IDs de tokens.
        """
        assert isinstance(s, str), "L'entrée doit être une chaîne de caractères"

        TIKTOKEN_MAX_ENCODE_CHARS = 400_000  # Limite maximale de caractères pour Tiktoken
        MAX_NO_WHITESPACES_CHARS = 25_000  # Limite maximale de caractères sans espaces

        # Découpage de la chaîne en sous-chaînes gérables
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)  # Ajout du token de début si nécessaire
        if eos:
            t.append(self.eos_id)  # Ajout du token de fin si nécessaire
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Décode une liste d'IDs de tokens en une chaîne de caractères.

        Args:
            t (List[int]): Liste des IDs de tokens.

        Returns:
            str: Chaîne décodée.
        """
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Découpe la chaîne `s` en sous-chaînes contenant au maximum `max_consecutive_slice_len`
        caractères consécutifs blancs ou non-blancs.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

class ChatFormat:
    def __init__(self, tokenizer: AvikamTokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        """
        Encode l'en-tête d'un message (rôle + contenu).

        Args:
            message (Message): Message à encoder.

        Returns:
            List[int]: Liste des IDs de tokens.
        """
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])  # Début d'en-tête
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))  # Encodage du rôle
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])  # Fin d'en-tête
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))  # Séparateur
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        """
        Encode un message complet (en-tête + contenu).

        Args:
            message (Message): Message à encoder.

        Returns:
            List[int]: Liste des IDs de tokens.
        """
        tokens = self.encode_header(message)  # Encodage de l'en-tête
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)  # Encodage du contenu
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])  # Fin de tour
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        """
        Encode une conversation complète pour le modèle.

        Args:
            dialog (Dialog): Conversation à encoder.

        Returns:
            List[int]: Liste des IDs de tokens.
        """
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])  # Début du texte
        for message in dialog:
            tokens.extend(self.encode_message(message))  # Encodage de chaque message
        # Ajout de l'en-tête d'un message assistant vide pour que le modèle complète
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens
