from termcolor import cprint

def print_status(text, color='blue'):
    cprint(text, color, attrs=[ 'reverse'])
