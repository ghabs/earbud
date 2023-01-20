import tkinter as tk
from tkinter.simpledialog import Dialog


class BotPanel(Dialog):
	"""
	The timeline panel that displays the bot notifications
	"""

	def __init__(self, master, title, bots):
		self.bots = bots
		super().__init__(parent=master, title=title)

	def toggle(self, bot):
		bot.active = not bot.active

	def body(self, master):
		padding = 10
		num = len(self.bots)
		x = 0
		y = 0
		while num > 0:
			c = tk.Checkbutton(master, padx=padding, pady=padding, variable=tk.BooleanVar(), onvalue=True, offvalue=False, command=lambda bot=self.bots[num-1]: self.toggle(bot),
							   text=str(self.bots[num-1]))
			if self.bots[num-1].active:
				c.select()
			c.grid(row=x, column=y)
			y += 1
			if y == num:
				y = 0
				x += 1
			num -= 1

	def validate(self):
		"""For confirming the user hit OK and not cancel"""
		self.result = self.bots
		return True
