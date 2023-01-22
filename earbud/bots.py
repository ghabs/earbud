from appdirs import user_data_dir
from typing import List
from langchain import OpenAI, PromptTemplate, LLMChain
import os
import json
import re

from earbud.datastructures import Transcript, Segment, BotConfig, Trigger, Action, TriggerType


class Bot():
    """
    The base class for all bots.
    #TODO: API representation for adding text to bots/determine global text datastructure
    """

    def __init__(self) -> None:
        self.active = True

    def _empty_segment(self, transcript: Transcript) -> bool:
        """
        Check if the most recent transcript segment is empty.
        """
        return transcript.peak().text == ""

    def __call__(self, text: str) -> str:
        """
        Run the bot on the text.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return a string representation of the bot.
        """
        if "name" in self.__dict__:
            return f"{self.name}"
        return f"{self.__class__.__name__}"

    def _to_json(self) -> str:
        """
        Return a json representation of the bot.
        """
        return json.dumps(self.__dict__)


class UserCreatedBot(Bot):
    """
    A bot that is created by a user.
    """

    def __init__(self, bot_config: BotConfig) -> None:
        self.bot_config = bot_config
        self.name = bot_config.name
        super().__init__()
        self._load()

    def _load(self):
        """
        Adds self variables from bot_config.
        """
        if self.bot_config.action.type == "prompt":
            # TODO: set this from global config
            self._llm = OpenAI()
            prompt = self.bot_config.action.action
            prompt += "{text}"
            self._prompt = PromptTemplate(
                input_variables=["text"], template=prompt)
            self.trigger = self.bot_config.trigger
            self.action = lambda text: self._llm(
                self._prompt.format(text=text))
        elif self.bot_config.action.type == "predefined":
            raise NotImplementedError
        elif self.bot_config.action.type == "match":
            r = re.compile(self.bot_config.action.action)
            self.action = lambda text: r.findall(text)
        elif self.bot_config.action.type == "substitute":
            r = re.compile(self.bot_config.action.action)
            #TODO add argument for substitution
            self.action = lambda text: r.sub("", text)
        else:
            raise NotImplementedError

    def __call__(self, transcript: Transcript) -> str:
        """
        Run the bot on the text.
        """
        # TODO: allow customization of amount of transcript
        if self.trigger(transcript.peak().text):
            return self.action(transcript.peak().text)
        else:
            return None


class BotCreator():
    """
    Create a new bot from a bot config.
    """

    def make_bot_config(self, name: str, trigger: str, trigger_type: str, action: str, action_type: str) -> Bot:
        """
        Accept a form input, parse, and create a bot config.
        """
        # TODO add validation and cleaning
        trigger = Trigger(input=[trigger.lower()], evaluation=TriggerType(trigger_type.lower()))
        action = Action(type=action_type.lower(), action=action)
        bot_config = BotConfig(name=name, trigger=trigger, action=action)
        return bot_config

    def store_bot_config(self, bot_config: BotConfig) -> None:
        """
        Store the bot config in user data directory.
        """
        user_data = user_data_dir("earbud", "earbud")
        bot_config_dir = os.path.join(user_data, "bot_configs")
        bot_config_path = os.path.join(
            bot_config_dir, f"{bot_config.name}.json")
        with open(bot_config_path, "w") as f:
            # TODO: check if bot config already exists
            bot_config_json = bot_config._to_json()
            json.dump(bot_config_json, f)

    def load_bot_config(self, name: str) -> BotConfig:
        """
        Load a bot config from user data directory.
        """
        d = json.loads(name)
        trigger = Trigger(**d["trigger"])
        action = Action(**d["action"])
        name = d["name"]
        bot_config = BotConfig(name=name, trigger=trigger, action=action)
        bot = self.create(bot_config)
        return bot

    def create(self, bot_config: BotConfig) -> Bot:
        """
        Create a bot from a bot config.
        """
        return UserCreatedBot(bot_config)


class KeywordBot(Bot):
    """
    A bot that searches for keywords in a sentence and returns the keyword.
    """

    def __init__(self, keywords: List[str]) -> None:
        self.name = "KeywordBot"
        self.keywords = keywords
        super().__init__()

    def __call__(self, text: str) -> str:
        """
        Run the bot on the text.
        """
        for keyword in self.keywords:
            if keyword in text:
                return keyword
        return None


class ConceptBot(Bot):
    def __init__(self, llm, k=1) -> None:
        self.name = "ConceptBot"
        self.llm = llm
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="From the following bullet points, identify 1 to 3 key concepts where more research could be useful:\n {text}")
        super().__init__()
        self.active = False

    def _concepts(self, segments: List[Segment]):
        segs = segments[-self.k:]
        text = " ".join([seg.text for seg in segs])
        prompt = self.prompt.format(text=text)
        return self.llm(prompt)

    def __call__(self, transcript: Transcript) -> str:
        """
        Run the bot on the text.
        """
        if not self._empty_segment(transcript):
            return None
        return self._concepts(transcript.segments)


class SummarizeBot(Bot):
    def __init__(self, llm, k=1, window=5) -> None:
        self.name = "SummarizeBot"
        self.llm = llm
        self.k = k
        self.window = window
        self.i = 0
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="Given the following in-progress meeting transcript, summarize the meeting so far in 1-2 sentences:\n {text}")
        super().__init__()

    def _summarize(self, segments: List[Segment]):
        segs = segments[-self.k:]
        text = " ".join([seg.text for seg in segs])
        prompt = self.prompt.format(text=text)
        return self.llm(prompt)

    def __call__(self, transcript: Transcript) -> str:
        """
        Run the bot on the text.
        """
        # Only summarize if the transcript is empty and the unsummarized transcript length is a multiple of self.tl
        segments = transcript.segments[self.i:]
        if self._empty_segment(transcript) or len([True for s in segments if s.text != ""]) < self.window:
            return None
        self.i = len(transcript.segments) - 1
        return self._summarize(transcript.segments)


if __name__ == "__main__":
    bot_config = BotConfig(name="test", trigger=Trigger(input=["test"], evaluation=TriggerType("contains")), action=Action(
        type="prompt", action="Given the following, return yes if the word is test\n {text}"))
    bot_creator = BotCreator()
    bot = bot_creator.create(bot_config)
    t = Transcript()
    t.segments.append(Segment(text="test", start=0, end=1))
    print(bot(t))
