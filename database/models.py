from dataclasses import dataclass
from typing import Optional


@dataclass
class QuizAttempt:
    timestamp: str
    score: int
    total: int
    topic: Optional[str] = None

    @property
    def percentage(self):
        if self.total == 0:
            return 0
        return round((self.score / self.total) * 100, 1)


@dataclass
class InterviewAttempt:
    timestamp: str
    question: str
    answer: str
    rating: Optional[int] = None
    feedback: Optional[str] = None


@dataclass
class DocumentRecord:
    timestamp: str
    filename: str
    word_count: int
    chunk_count: int