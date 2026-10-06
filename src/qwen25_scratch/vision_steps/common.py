"""Small utilities shared by the independent vision exercises."""


class ExerciseIncomplete(NotImplementedError):
    """Raised at a tensor operation that the student must implement."""


def print_incomplete(error: ExerciseIncomplete, filename: str) -> None:
    """Print a friendly next action while keeping an exercise runnable."""
    print(f"\nINCOMPLETE: {error}")
    print(f"Edit {filename}, replace its TODO exception, and run this module again.")
