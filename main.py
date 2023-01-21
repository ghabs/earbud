from appdirs import user_log_dir
import os
import logging

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
	log_dir = user_log_dir("Earbud")
	log_file = os.path.join(log_dir, "earbud.log")
	logging.basicConfig(
		filename=log_file,
		level=logging.DEBUG,
		format="[%(asctime)s] %(module)s.%(funcName)s:%(lineno)d %(levelname)s -> %(message)s")
	
	from earbud.gui import GUI
	app = GUI()
	app.mainloop()

