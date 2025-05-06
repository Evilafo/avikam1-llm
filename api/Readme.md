# **Avikam1 LLM API Documentation**

## Description

L'API **Avikam1 LLM** est une interface de chat permettant aux utilisateurs d'interagir avec l'IA d'Evilafo AI. Cette API fournit des réponses générées par le modèle de langage **Avikam1 LLM** en fonction des messages envoyés par l'utilisateur. L'API maintient également un contexte de conversation pour rendre les réponses plus adaptées et contextuelles.

---

## Prérequis

* **Token Bearer** : Un token d'authentification est nécessaire pour accéder à l'API.
* **Outils recommandés** : Utilise un outil comme **Postman**, **Insomnia**, ou **cURL** pour envoyer des requêtes HTTP.

---

## Démarrer le serveur

### Démarrage local

Pour démarrer l'API en local, utilise la commande suivante :

```bash
uvicorn main:app --reload
```

L'API sera accessible à l'adresse [https://apiavikam1llm.evilafo.xyz](https://apiavikam1llm.evilafo.xyz).

Le token pour les test est : 
```bash 
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJleHBpcmF0aW9uX3RpbWUiOjE2Mzg3Nzk5OTgsImlhdCI6MTYzODc3OTk5OH0.srV5-ExvTtvUPnRwncbM4ZX-hQxjvYkAw-H5Pz56XjE
```

---

## Authentification

L'API utilise un **Bearer Token** pour authentifier les requêtes. Ce token doit être inclus dans l'en-tête `Authorization` de chaque requête HTTP.

### Exemple d'en-tête de requête :

```
Authorization: Bearer <ton_token>
```

---

## Endpoints

### 1. **`/chat`** : Envoyer un message et obtenir une réponse

#### Description

Cet endpoint permet à l'utilisateur d'envoyer un message et d'obtenir une réponse générée par **Avikam1 LLM**. L'API maintient un historique des messages pour contextualiser les réponses.

#### Méthode

* **POST**

#### URL

```
POST /chat
```

#### Paramètres

* **Authorization (Header)** : Un token Bearer est requis pour authentifier la requête.
* **Content-Type (Header)** : `application/json`

#### Corps de la requête

Tu dois envoyer un objet JSON contenant le message de l'utilisateur.

Exemple de requête JSON :

```json
{
  "message": "Quel temps fait-il aujourd'hui ?"
}
```

* **message** : Le texte de la question ou de la demande que tu veux envoyer à l'API.

#### Réponse

La réponse sera un objet JSON contenant la réponse générée par l'IA.

Exemple de réponse :

```json
{
  "id": 1,
  "content": "Il fait beau aujourd'hui. Quel temps fait-il dans votre région ?"
}
```

* **id** : L'ID de la réponse générée (identifiant unique).
* **content** : La réponse générée par le modèle.

---

## Scénarios d'exemple

### Exemple 1 : Demande simple

* **Requête** :

  ```json
  {
    "message": "Quel temps fait-il aujourd'hui ?"
  }
  ```

* **Réponse** :

  ```json
  {
    "id": 1,
    "content": "Il fait beau aujourd'hui. Quel temps fait-il dans votre région ?"
  }
  ```

### Exemple 2 : Demande sur l'identité de l'assistant

Si l'utilisateur demande "Qui es-tu ?", l'API renverra une réponse définie dans le contexte, sans mentionner d'informations génériques.

* **Requête** :

  ```json
  {
    "message": "Qui es-tu ?"
  }
  ```

* **Réponse** :

  ```json
  {
    "id": 2,
    "content": "Je suis **Evilafo AI**, un assistant d'intelligence artificielle conçu pour vous aider avec vos questions et tâches. Comment puis-je vous aider ?"
  }
  ```

---

## Sécurisation de l'API avec Token

L'accès à l'API est protégé par un token Bearer. Tu dois inclure ce token dans les en-têtes de ta requête.

### Exemple avec **cURL** :

```bash
curl -X POST "https://apiavikam1llm.evilafo.xyz/chat" \
  -H "Authorization: Bearer <ton_token_ici>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Quel temps fait-il aujourd'hui ?"}'
```

### Exemple avec **Postman** :

1. Sélectionne la méthode `POST`.

2. Utilise l'URL `https://apiavikam1llm.evilafo.xyz/chat`.

3. Dans l'onglet **Authorization**, sélectionne **Bearer Token** et entre ton token.

4. Dans l'onglet **Body**, sélectionne **raw** et choisis le type **JSON**. Ensuite, entre le message :

   ```json
   {
     "message": "Quel temps fait-il aujourd'hui ?"
   }
   ```

5. Clique sur **Send** pour envoyer la requête.

---

## Exemple d'Appel à l'API

Imaginons que tu veuilles envoyer un message comme "Quel est le temps aujourd'hui ?". Voici comment procéder :

### 1. **URL** : `https://apiavikam1llm.evilafo.xyz/chat`

### 2. **Méthode HTTP** : `POST`

### 3. **En-têtes** :

* `Authorization: Bearer <ton_token>`
* `Content-Type: application/json`

### 4. **Corps de la requête** :

```json
{
  "message": "Quel est le temps aujourd'hui ?"
}
```

### 5. **Réponse attendue** :

```json
{
  "id": 1,
  "content": "Il fait beau aujourd'hui. Quel temps fait-il dans votre région ?"
}
```

---


## Licence

Ce projet est sous la licence **MIT**. Tu peux l'utiliser, le modifier et le distribuer selon les termes de cette licence.

---

## Contact

Si tu rencontres des problèmes ou as des questions, n'hésite pas à créer une **issue** sur GitHub ou à nous contacter directement via l'adresse suivante : **[evil2846gmail.com](mailto:evil2846@gmail.com)**.

