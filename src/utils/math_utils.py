import sympy

def evaluate_equation(equation: str):
    """
    Évalue une équation mathématique donnée sous forme de chaîne.
    Args:
        equation (str): Équation à évaluer (e.g., "x**2 + 3*x + 2").
    Returns:
        float: Résultat de l'évaluation.
    """
    try:
        expr = sympy.sympify(equation)
        result = expr.evalf()
        return result
    except Exception as e:
        raise ValueError(f"Erreur lors de l'évaluation de l'équation : {e}")
