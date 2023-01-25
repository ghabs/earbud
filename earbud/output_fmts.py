import os
from appdirs import user_data_dir

user_data_folder = user_data_dir("earbud", "earbud")
ouput_fmts = os.path.join(user_data_dir("earbud", "earbud"), "output_fmts")
if not os.path.exists(ouput_fmts):
    os.makedirs(ouput_fmts)

def list_output_fmts():
    return [f for f in os.listdir(ouput_fmts) if f.endswith(".md")]

def get_output_fmt(name):
    with open(os.path.join(ouput_fmts, name)) as f:
        return f.read()

def get_output_fmts(fmts):
    return [{"name": name.rsplit('.', 1)[0], "value": get_output_fmt(name)} for name in fmts]

def user_output_fmts():
    return get_output_fmts(list_output_fmts())

def save_output_fmt(name, value):
    with open(os.path.join(ouput_fmts, name + ".md"), "w") as f:
        f.write(value)


if __name__ == "__main__":
    print(get_output_fmts(list_output_fmts()))