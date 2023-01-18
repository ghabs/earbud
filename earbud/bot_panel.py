class BotPanel():
    """
    The timeline panel that displays the bot notifications
    """
    def __init__(self) -> None:
        self.time_to_live = 5
        self.collapsed = False
        self.displayed_bots = []