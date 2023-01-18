class Bot():
    """
    The base class for all bots.
    #TODO: API representation for adding text to bots/determine global text datastructure
    """
    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> str:
        """
        Run the bot on the text.
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """
        Return a string representation of the bot.
        """
        return f"{self.__class__.__name__}()"


class BotManager():
    """
    Hold bots and manage them for the main GUI
    GUI: Display the different bots and allow a user to set the params in settings panel
    """
    pass