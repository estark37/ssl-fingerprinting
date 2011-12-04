from webbrowser import open_new_tab
import os, sys, time
import subprocess
from random import choice, randint


#sites = ["http://google.com", "http://www.facebook.com", "http://yahoo.com",
#         "http://amazon.com", "http://wikipedia.org", "http://twitter.com", "http://ebay.com",
#         "http://blogger.com", "http://craigslist.org", "http://linkedin.com", "http://live.com",
#         "http://go.com", "http://msn.com", "http://bing.com", "http://espn.go.com", "http://cnn.com",
#         "http://apple.com", "http://paypal.com", "http://aol.com"]

sites = ["https://encrypted.google.com", "https://www.facebook.com", "https://twitter.com",
         "https://www.bankofamerica.com", "https://online.citibank.com", "https://www.box.net",
         "https://www.dropbox.com", "https://www.torproject.org"]

bg_sites = []

num_visits = 60

def open_browser():
    child = subprocess.Popen("/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome about:blank --incognito", shell=True)
    return child

def close_browser(browser):
    browser.terminate()

def start_tcpdump(name):
    child = subprocess.Popen("tcpdump -i en1 -w %s port 80 or port 443"%name, shell=True)
    return child

def end_tcpdump(dump):
    dump.terminate()

def safe(site):
    return site.find("adult") == -1 and site.find("xxx") == -1 and site.find("porn") == -1 and site.find("sex") == -1

def load_bg_sites():
    for line in open("top-1m.csv"):
        # try to be safe...
        if safe(line):
            bg_sites.append(line.split(",")[1][:-1])
        if len(bg_sites) >= 5000:
            return

def generate_data(bg_visit, data_root):
    for site in sites:
        print "Site: %s"%site
        for i in range(num_visits):
            dump = start_tcpdump("%s%s_%d.dat"%(data_root, site.split("//")[1], i))
            browser = open_browser()
            bg = choice(bg_sites)
            pid = os.fork()
            if pid:
                print "Opening %s"%site
                time.sleep(15)
                close_browser(browser)
                end_tcpdump(dump)
            else:
                time.sleep(7)
                open_new_tab(site)
                if (bg_visit):
                    open_new_tab("http://%s"%bg)
                sys.exit()
            time.sleep(6)

load_bg_sites()
# single visit data
#generate_data(False, "data/single_visits/")
# multiple visits data
generate_data(True, "data/mult_visits/")



