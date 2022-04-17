class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



from termcolor import colored
def _print(string, p=None):
  if not p:
    print(string)
    return
  pre = f"{bcolors.ENDC}"
  
  if "bold" in p.lower():
    pre += bcolors.BOLD
  elif "underline" in p.lower():
    pre += bcolors.UNDERLINE
  elif "header" in p.lower():
    pre += bcolors.HEADER
      
  if "warning" in p.lower():
    pre += bcolors.WARNING
  elif "error" in p.lower():
    pre += bcolors.FAIL
  elif "ok" in p.lower():
    pre += bcolors.OKGREEN
  elif "info" in p.lower():
    if "blue" in p.lower():
      pre += bcolors.OKBLUE
    else: 
      pre += bcolors.OKCYAN

  print(f"{pre}{string}{bcolors.ENDC}")



import yaml
def load_config(config_filepath):
  try:
    with open(config_filepath, 'r') as file:
      config = yaml.safe_load(file)
      return config
  except FileNotFoundError:
    _print(f"Config file not found! <{config_filepath}>", "error_bold")
    exit(1)