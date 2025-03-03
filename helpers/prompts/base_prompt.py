from abc import ABC, abstractmethod
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class BasePrompt(ABC):
    @abstractmethod
    def _get_system_prompt(self) -> SystemMessagePromptTemplate:
        pass

    @abstractmethod
    def _get_human_prompt(self) -> HumanMessagePromptTemplate:
        pass

    def get_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [self._get_system_prompt(), self._get_human_prompt()]
        )
