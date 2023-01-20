import tkinter as tk
from tkinter.simpledialog import Dialog
from tkinter import ttk

from bots import BotCreator, BotConfig


class BotPanel(Dialog):
	"""
	The timeline panel that displays the bot notifications
	"""

	def __init__(self, master, title, bots):
		self.bots = bots
		super().__init__(parent=master, title=title)

	def toggle(self, bot):
		bot.active = not bot.active
	
	def create_bot(self):
		w = CreateBotPanel(self, "Create Bot")
		if w.result:
			self.bots.append(w.result)
			#TODO: get window to update
			self.update()

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
		
		self.create_bot = ttk.Button(master, text="Create Bot", command=self.create_bot)
		self.create_bot.grid(row=x+1, column=0)

	def validate(self):
		"""For confirming the user hit OK and not cancel"""
		self.result = self.bots
		return True

class CreateBotPanel(Dialog):
	def __init__(self, parent, title) -> None:
		super().__init__(parent, title)
	
	def body(self, master) -> None:
		nameBot = ttk.Label(master, text="Name of Bot")
		nameBot.grid(row=0, column=0)
		self.name_of_bot = ttk.Entry(master)
		self.name_of_bot.grid(row=0, column=1)

		triggerLabel = ttk.Label(master, text="Trigger")
		triggerLabel.grid(row=1, column=0)
		self.trigger_var = tk.StringVar()
		self.trigger = ttk.OptionMenu(master, self.trigger_var, "", "Text", "Time")
		self.trigger.grid(row=1, column=1)
		
		def trigger_callback():
			if self.trigger_var.get() == "Text":
				self.trigger_input = ttk.Entry(master)
				self.trigger_input.grid(row=2, columnspan=2)
			else:
				self.trigger_input = ttk.Entry(master)
				self.trigger_input.grid(row=2, columnspan=2)

		self.trigger_var.trace("w", lambda *args: trigger_callback())

		actionLabel = ttk.Label(master, text="Action")
		actionLabel.grid(row=3, column=0)
		self.action_var = tk.StringVar()
		self.action = ttk.OptionMenu(master, self.action_var, "", "Predefined", "Prompt")
		self.action.grid(row=3, column=1)

		def action_callback():
			if self.action_var.get() == "Predefined":
				self.action_input = ttk.Entry(master)
				self.action_input.grid(row=4, columnspan=2)
			else:
				self.action_input = ttk.Entry(master)
				self.action_input.grid(row=4, columnspan=2)
		
		self.action_var.trace("w", lambda *args: action_callback())

	def validate(self):
		bot_creator = BotCreator()
		bot_config = bot_creator.make_bot_config(self.name_of_bot.get(), self.trigger_input.get(), self.trigger_var.get(), self.action_input.get(), self.action_var.get())
		#maybe make a singleton factory class for creating bots
		self.result = bot_creator.create(bot_config)
		return True




